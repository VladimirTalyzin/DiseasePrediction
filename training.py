from enum import Enum
from os import path

from joblib import dump, load
from keras.layers import Dense, Dropout
from keras.models import load_model
# noinspection PyUnresolvedReferences
from keras.optimizers import Adam, SGD, RMSprop
from numpy import zeros, argmax
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tqdm import tqdm
from tqdm.keras import TqdmCallback

# типы движков для работы моделей
# сейчас поддерживаются SCIKIT_LEARN или KERAS
class TrainingEngine(Enum):
    SCIKIT_LEARN = 1
    KERAS = 2


# класс тренировки моделей
class Training:
    __model = None
    __scalerInput = MinMaxScaler()
    __scalerOutput = MinMaxScaler()
    __predict = None
    __withScale = True
    __engine = TrainingEngine.SCIKIT_LEARN
    __columnsMode = False


    # engine - тип движка обучения SCIKIT_LEARN или KERAS
    # filename - путь к файлу модели, без указания расширения, так как оно зависит от типа модели
    # withScale - автоматическое масштабирование данных в интервал [0, 1]
    def __init__(self, engine, filename, withScale, columnsMode):
        self.__withScale = withScale
        self.__engine = engine
        self.__columnsMode = columnsMode

        # если файл модели указан, то загружаем модель
        if filename is not None:
            self.__loadModel(filename, columnsMode)


    # inputData - данные пациентов
    # outputData - данные о заболеваниях для обучения
    # filename - имя файла для сохранения модели. Также используется за загрузки модели колонки, если задано пропустить обучение колонки
    # classWeights - начальные веса классов для каждой из колонок
    # классы нужны, так как при не очень точных данных и редких случаях заболевания,
    # оптимизатор справедливо полагает, что вероятнее всего, заболевания вообще не будет,
    # чтобы избежать такого результата, повышаем вес класса заболевания

    # tolerances - допустимое статистическое отклонение данных по колонке предсказания и реального значения
    # targetAccuracies - допустимое максимальное среднеквадратичное отклонение для каждой колонки
    # epochs - количество эпох обучения
    # searchBestTries - сколько нужно накопить подходящих моделей для выбора самой хорошей по метрике оптимизации
    # skips - возможность пропустить какие-либо из колонок. В этом случае данные будут прочитаны из имеющегося файла модели
    # metricaName - метрика выбора самой оптимальной модели
    # metricaFun - функция выбора самой оптимальной модели min или max
    def train(self, inputData, outputData, filename, classWeights, tolerances, targetAccuracies, epochs, searchBestTries, skips, metricaName, metricaFun):
        # если при создании модели выбрано масштабирование, то необходимо провести масштабирование
        # с помощью MinMaxScaler. при этом надо сохранить данные масштабирования, так как
        # минимум и максимум данных для обучения и для предсказания не совпадают
        if self.__withScale:
            self.__scalerInput.fit(inputData)
            self.__scalerOutput.fit(outputData)

            inputData  = self.__scalerInput.transform(inputData)
            outputData = self.__scalerOutput.transform(outputData)

        # вызвать функцию обучения в соответствии с выбранным движком модели
        if self.__engine == TrainingEngine.SCIKIT_LEARN:
            self.__sciKitTraining(inputData, outputData, classWeights)

        elif self.__engine == TrainingEngine.KERAS:
            self.__kerasTraining(inputData, outputData, filename, classWeights, tolerances, targetAccuracies, epochs, searchBestTries, skips, metricaName, metricaFun)


    # обучение KERAS
    def __kerasTraining(self, inputData, outputData, filename, classWeights, tolerances, targetAccuracies, epochs, searchBestTries, skips, metricaName, metricaFun):
        # сколько колонок у входных данных
        lengthInput = len(inputData[0])
        # сколько колонок у выходных данных
        lengthOutput = len(outputData[0])

        # режим разных моделей для каждой из колонок результата
        if self.__columnsMode:
            self.__model = []

            # подсчёт минимального среднеквадратичного отклонения для каждой колонки
            minErrors = [100] * lengthOutput
            # последнее среднеквадратичное отклонение для каждой колонки
            lastErrors = [100] * lengthOutput

            # выполняем обучение для каждой колонки
            for output in range(lengthOutput):

                # получаем настройки для колонки по её номеру
                classWeight = classWeights[output]
                tolerance = tolerances[output]
                targetAccuracy = targetAccuracies[output]
                epochsCount = epochs[output]
                searchBestTry = searchBestTries[output]
                skip = skips[output]

                # если указано пропустить колонку, то не выполняем обучение, а загружаем из файла
                if skip:
                    while path.isfile(filename + "_" + str(output) + ".h5"):
                        self.__model.append(load_model(filename + "_" + str(output) + ".h5"))
                    continue

                # словарь, где будут накапливаться модели, с ключом, равным значению metrica
                # затем из этого словаря будет выбрано значение с metricaFun
                bestModels = {}
                bestModelsCounter = 0

                # выполняем обучение, пока в словаре bestModels не накопится заданное количество моделей для выбора лучшей
                while bestModelsCounter < searchBestTry:
                    model = None

                    wrongWeight = True
                    tries = 0
                    metricaValue = None

                    # если при обучении количество 1 отклонилось от количества 1 в обучающей выборке более чем на tolerance, то повторяем обучение
                    while wrongWeight:

                        # создаём модель Keras
                        # параметры ниже получены опытным путём
                        model = keras.Sequential([
                            # на первом скрытом слое 40 нейронов с активацией relu
                            # использовать на верхнем уровне кусочно-линейную функцию логично,
                            # чтобы максимально сохранить входные данные, выполняя минимум преобразования
                            # исходных данных, которые уже хорошо подготовлены
                            Dense(40, input_shape = (lengthInput,), activation = "relu"),

                            # предполагаю, что не все из исходных данных имеют реальное влияние на результат
                            # поэтому с вероятностью 40% на этом уровне случайные веса будут обнулены
                            Dropout(0.4),

                            # ещё один скрытый слой с более естественной для нейронных сетей функцией сигмоида
                            Dense(20, activation = "sigmoid"),

                            # снова обнуляем случайные веса с вероятностью 30%
                            Dropout(0.3),

                            # выходной слой содержит два нейрона, так как возможны два класса - есть заболевание или нет
                            # функция активации при заданной работе с классами возможна только softmax
                            Dense(2, activation = "softmax")
                        ])

                        # noinspection SpellCheckingInspection
                        # указываем подобранную опытным путём функцию поиска оптимального решения
                        # и функцию подсчёта потери
                        # указываем метрики для вывода информации обучения
                        model.compile(
                            optimizer = Adam(learning_rate = 0.001), # learning_rate = 0.001
                            # optimizer = Nadam(learning_rate = 0.001),
                            # optimizer = RMSprop(learning_rate = 0.001),
                            # optimizer = SGD(learning_rate = 0.01),
                            # loss = 'binary_crossentropy',
                            # loss = "hinge",
                            loss = "categorical_crossentropy",
                            # loss = "mean_squared_error",
                            # loss = "sparse_categorical_entropy",
                            metrics=[
                                keras.metrics.MeanSquaredError(),
                                #keras.metrics.BinaryCrossentropy(),
                                keras.metrics.CategoricalAccuracy()
                                #keras.metrics.Precision(),
                                #keras.metrics.AUC()
                            ]
                        )

                        print("Learning output", output, "try", tries + 1, " best collected", bestModelsCounter)

                        # получаем выходные данные по текущей колонке
                        outputColumn = outputData[:, output]
                        # переводим значения в двумерный тензор, где [1, 0] ставится для 0 и [0, 1] для 1
                        outputColumnCategories = keras.utils.to_categorical(outputColumn, 2)

                        metrics = None

                        # возможно несколько раз обучить модель на разном разбиении исходных данных
                        for state in range(1):
                            # разбить обучающие данные в пропорции 0.85/0.15
                            X_train, X_valid, Y_train, Y_valid = train_test_split(inputData, outputColumnCategories, test_size = 0.15,
                                                                                  random_state = output * (state + 1) * (bestModelsCounter + 1))

                            # запустить обучение на 0.85 исходных данных, а 0.15 исходных данных используется как проверка качества обучения
                            # также задаётся вес класса 1
                            # обучений за раз равно количеству входных значений
                            # процесс обучения выводим в красивом ProgressBar
                            metrics = model.fit(X_train, Y_train, epochs = int(epochsCount / 1), batch_size = lengthInput, validation_data = (X_valid, Y_valid),
                                      class_weight = classWeight,
                                      verbose = 0,
                                      callbacks = [TqdmCallback(verbose = 0)])

                        #exit(0)
                        #metrics = model.fit(inputData, outputColumnCategories, epochs = epochsCount, batch_size = lengthInput,
                        #          class_weight = classWeight,
                        #          verbose = 0,
                        #          callbacks = [TqdmCallback(verbose = 0)],
                        #          validation_split = 0.2)

                        # получаем и сохраняем среднеквадратичное отклонение
                        lastErrors[output] = metrics.history["mean_squared_error"][-1]

                        # находим минимальное среднеквадратичное отклонение
                        if minErrors[output] > lastErrors[output]:
                            minErrors[output] = lastErrors[output]

                        # получаем значение заданной метрики
                        metricaValue = metrics.history[metricaName][-1]

                        # считаем пропорцию единиц в исходных данных
                        inputWeight = outputColumn.sum() / len(inputData)
                        # считаем пропорцию единиц в предсказанных данных
                        calculatedWeight = argmax(model.predict(inputData), axis=-1).sum() / len(inputData)

                        wrongWeight = False
                        # запоминаем предыдущий вес класса для вывода в строке информации
                        oldClassWeight = classWeight[1] if classWeight is not None else None

                        # если разница между пропорцией исходных данных и предсказанных данных больше заданного значения tolerance для колонки
                        if abs(calculatedWeight - inputWeight) > tolerance:
                            # изменяем значение веса класса в большую или меньшую сторону
                            # важно, чтобы изменение в большую сторону не было точно равно изменению в меньшую сторону
                            # чтобы избежать бесконечных циклов
                            if classWeight is None:
                                wrongWeight = True

                            elif calculatedWeight - inputWeight > tolerance:
                                    classWeight[1] -= 0.05
                                    wrongWeight = True

                            elif calculatedWeight - inputWeight < -tolerance:
                                classWeight[1] += 0.06
                                wrongWeight = True

                        # выводим собранную информацию о весах классов
                        print("Calculated weight: " + format(calculatedWeight, '.3f') + " expected weight: " + format(inputWeight, '.3f') +
                              ((" Class weight: " + format(oldClassWeight, '.2f') + " => " + format(classWeight[1], '.2f'))
                                    if oldClassWeight is not None and oldClassWeight != classWeight[1] else ""))

                        # если пропорция была за пределами заданных значений, или среднеквадратичное отклонение выше заданной,
                        # то повторяем обучение заново
                        if not wrongWeight and targetAccuracy < lastErrors[output]:
                            wrongWeight = True

                        # выводим среднеквадратичное отклонение и ожидаемое максимальное среднеквадратичное отклонение
                        print("Calculated loss: " + format(lastErrors[output], '.3f') + " expected loss: " + format(targetAccuracy, '.3f'))
                        tries += 1

                    # если модель прошла все проверки, то накапливаем в словаре хороших моделей
                    # ключ - значение выбранной метрики
                    bestModels[metricaValue] = model
                    bestModelsCounter += 1

                # выводим минимальную и максимальную метрику, а также указанную функцию метрики
                minMetrica = min(bestModels.keys())
                maxMetrica = max(bestModels.keys())
                selectedMetrica = metricaFun(bestModels.keys())
                print("Found min " + metricaName + ": " + format(minMetrica, '.3f') + " max: " + format(maxMetrica, '.3f') + " selected: " + format(selectedMetrica, '.3f'))

                # выбираем самую лучшую модель для колонки
                model = bestModels[selectedMetrica]

                # сохраняем модель колонки
                self.__model.append(model)
                model.save(filename + "_" + str(output) + ".h5")

            print("Min errors: ", minErrors)
            print("Last errors: ", lastErrors)

        else:
            # работа в режиме общей модели для всех колонок
            # эта ветка показала плохие результаты и не прорабатывалась
            model = keras.Sequential([
                Dense(40, activation = "sigmoid"),
                Dropout(0.5),
                Dense(lengthOutput, activation = "sigmoid")
            ])

            # noinspection SpellCheckingInspection
            model.compile(
                optimizer = Adam(learning_rate = 0.01),  # learning_rate = 0.01 # 0.0274, 0.0052
                # optimizer = Nadam(learning_rate=0.01), # 0.0332
                # optimizer = RMSprop(learning_rate = 0.01),  # 0.0255, 0.0056
                # optimizer = SGD(), #0.03683
                loss = 'binary_crossentropy',
                # loss = keras.losses.BinaryCrossentropy(),
                # loss = "hinge",
                # loss="categorical_crossentropy",
                # loss = "mean_squared_error",
                # loss = "sparse_categorical_entropy",
                metrics=[
                    keras.metrics.BinaryCrossentropy(),
                    keras.metrics.CategoricalAccuracy()
                    # keras.metrics.Precision(),
                    # keras.metrics.AUC()
                ]
            )

            for _ in range(5):
                X_train, X_valid, Y_train, Y_valid = train_test_split(inputData, outputData, test_size=0.2)
                model.fit(X_train, Y_train, epochs=50, batch_size=lengthInput, validation_data=(X_valid, Y_valid),
                          #class_weight={0: None, 1: {0: 1, 1: 6}, 2: {0: 1, 1: 4}, 3: {0: 1, 1: 4}, 4: {0: 1, 1: 6}}[output],
                          verbose=0,
                          callbacks=[TqdmCallback(verbose=0)])

            model.fit(inputData, outputData, epochs=100, batch_size=lengthInput,
                      #class_weight={0: None, 1: {0: 1, 1: 6}, 2: {0: 1, 1: 4}, 3: {0: 1, 1: 4}, 4: {0: 1, 1: 6}}[output],
                      verbose=0,
                      callbacks = [TqdmCallback(verbose=0)])

            self.__model = model

            # self.__model.fit(inputData, outputData, batch_size = 64, epochs = 1000)
            # testLoss, testAccuracy = self.__model.evaluate(inputData, outputData, batch_size = 64)
            # print("Test loss: " + testLoss, "Test accuracy: " + testAccuracy)


    # поиск оптимального значения с помощью SCIKIT-LEARN
    def __sciKitTraining(self, inputData, outputData, classWeights):
        lengthOutput = len(outputData[0])
        self.__model = []

        # модели строятся отдельно по каждой из колонок
        for output in range(lengthOutput):
            # используем специальный класс для поиска классификации - SGDClassifier
            model = SGDClassifier(loss ='hinge', penalty = 'l2', alpha = 0.001, random_state = 42, max_iter = 500000, tol = None,
                                  class_weight = classWeights[output])


            outputColumn = outputData[:, output]
            model.fit(inputData, outputColumn)
            print("Learn score output " + str(output) + " = ", model.score(inputData, outputColumn))
            self.__model.append(model)


    # получить предсказание модели
    def predict(self, inputData, columnsCount):
        if self.__withScale:
            inputData = self.__scalerInput.transform(inputData)

        outputData = None
        if self.__engine == TrainingEngine.SCIKIT_LEARN:
            if type(self.__model) is list:
                lengthInput = len(inputData)
                outputData = zeros((lengthInput, columnsCount))
                for index, model in enumerate(self.__model):
                    output = model.predict(inputData)
                    outputData[:, index] = output

            else:
                # noinspection PyUnresolvedReferences
                outputData = self.__model.predict(inputData)

        elif self.__engine == TrainingEngine.KERAS:
            if type(self.__model) is list:
                lengthInput = len(inputData)
                outputData = zeros((lengthInput, columnsCount))
                for index, model in tqdm(enumerate(self.__model), leave=False):
                    output = model.predict(inputData)
                    outputData[:, index] = argmax(output, axis=-1)

            else:
                # noinspection PyUnresolvedReferences
                outputData = self.__model.predict(inputData)

        if self.__withScale:
            outputData = self.__scalerOutput.inverse_transform(outputData)

        self.__predict = outputData


    def getPredictData(self, ids):
        result = []

        counter = 0
        for row in self.__predict:
            rowArray = [ids[counter]]
            counter += 1
            for value in row:
                if str(type(value)) == "<class 'numpy.ndarray'>":
                    roundedValue = round(value[0])
                else:
                    roundedValue = round(value)

                if roundedValue < 0:
                    roundedValue = 0

                if roundedValue > 1:
                    roundedValue = 1

                rowArray.append(roundedValue)

            result.append(rowArray)

        return result


    # сохранить модели в файл, в зависимости от выбранного движка и режима работы
    # списка разных моделей для каждой колонки или одной общей модели
    def saveModel(self, filename):
        if self.__model is not None:
            if self.__engine == TrainingEngine.SCIKIT_LEARN:
                if type(self.__model) is list:
                    for index, model in enumerate(self.__model):
                        with open(filename + "_" + str(index) + ".scikit", 'wb') as saveFile:
                            dump(model, saveFile)
                else:
                    with open(filename + ".scikit", 'wb') as saveFile:
                        dump(self.__model, saveFile)


            elif self.__engine == TrainingEngine.KERAS:
                    if type(self.__model) is list:
                        for index, model in enumerate(self.__model):
                            model.save(filename + "_" + str(index) + ".h5")
                    else:
                        # noinspection PyUnresolvedReferences
                        self.__model.save(filename + ".h5")

            if self.__withScale:
                with open(filename + ".input", 'wb') as saveFile:
                    dump(self.__scalerInput, saveFile)

                with open(filename + ".output", 'wb') as saveFile:
                    dump(self.__scalerOutput, saveFile)


    # загрузка модели. private-метод, так как вызывается из конструктора
    # в зависимости от выбранного движка и режима работы
    # разных моделей для каждой колонки или одной общей модели
    def __loadModel(self, filename, columnsMode):
        if self.__engine == TrainingEngine.SCIKIT_LEARN:
            if path.isfile(filename + ".scikit"):
                with open(filename + ".scikit", 'rb') as loadFile:
                    self.__model = load(loadFile)

        elif self.__engine == TrainingEngine.KERAS:
            if columnsMode:
                self.__model = []
                counter = 0
                while path.isfile(filename + "_" + str(counter) + ".h5"):
                    self.__model.append(load_model(filename + "_" + str(counter) + ".h5"))
                    counter += 1

            else:
                if path.isfile(filename + ".h5"):
                    self.__model = load_model(filename + ".h5")

        if self.__withScale:
            if path.isfile(filename + ".input"):
                with open(filename + ".input", 'rb') as loadFile:
                    self.__scalerInput = load(loadFile)

            if path.isfile(filename + ".output"):
                with open(filename + ".output", 'rb') as loadFile:
                    self.__scalerOutput = load(loadFile)