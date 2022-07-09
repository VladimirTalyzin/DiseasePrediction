from config import trainFile, checkFile, resultFile, resultCheckTrainingFile, columns
from convert import Convert
from training import Training, TrainingEngine
from output import Output

# считываем данные из файла для тренировки модели и преобразуем в числовые данные
convert = Convert(filename = trainFile, withCheckData = True)

# создаём класс для обучения модели
training = Training(
    # не указываем файл для загрузки модели, так как это новое обучение
    filename = None,
    # выбираем движок для обучения. Лучше, конечно, оказался KERAS
    engine = TrainingEngine.KERAS,
    # автоматическое масштабирование данных. Отключено, так как данные уже масштабированы вручную
    withScale = False,
    # режим разных моделей для каждой из колонок
    columnsMode = True)

# запускаем обучение
training.train(
    # данные для обучения
    convert.getTrainingData(),
    # данные о заболеваниях
    convert.getCheckData(),
    # файл, куда будет сохранена модель или модели в режиме разных моделей для каждой колонки
    "models/diseases",
    # начальные веса классов для каждой из колонок
    # классы нужны, так как при не очень точных данных и редких случаях заболевания,
    # оптимизатор справедливо полагает, что вероятнее всего, заболевания вообще не будет,
    # чтобы избежать такого результата, повышаем вес класса заболевания
    classWeights = {0: None, 1: {0: 1, 1: 4.20}, 2: {0: 1, 1: 3.21}, 3: {0: 1, 1: 2.93}, 4: {0: 1, 1: 4.95}},
    # допустимое статистическое отклонение данных по колонке предсказания и реального значения
    tolerances = {0: 0.03, 1: 0.007, 2: 0.010, 3: 0.010, 4: 0.010},
    # допустимое максимальное среднеквадратичное отклонение для каждой колонки
    targetAccuracies = {0: 0.187, 1: 0.083, 2: 0.126, 3: 0.117, 4: 0.148},
    # количество эпох обучения
    epochs = {0: 40, 1: 40, 2: 40, 3: 40, 4: 40},
    # сколько нужно накопить подходящих моделей для выбора самой хорошей по метрике оптимизации
    searchBestTries = {0: 20, 1: 30, 2: 30, 3: 30, 4: 30},
    # возможность пропустить какие-либо из колонок. В этом случае данные будут прочитаны из имеющегося файла модели
    skips = {0: False, 1: False, 2: False, 3: False, 4: False},
    # метрика выбора самой оптимальной модели
    metricaName="val_categorical_accuracy",
    # функция выбора самой оптимальной модели min или max
    metricaFun = max
)

# выполнить предсказание для тестирования качества моделей. Предсказанные данные не должны
# сильно отличаться от тех, что реально указаны в файле тренировки
training.predict(convert.getTrainingData(), len(columns) - 1)
# записать полученное предсказание в файл resultCheckTrainingFile. По нему можно вручную проверить качество работы моделей
Output(data = training.getPredictData(convert.getIDs()), resultFileName = resultCheckTrainingFile, columns = columns).output()

# подготовить данные из файла для предсказания заболеваний
convert = Convert(filename = checkFile, withCheckData = False)
# предсказать данные по заболеваниям
training.predict(convert.getTrainingData(), len(columns) - 1)
# записать полученные предсказания в файл resultFile. Этот файл надо оправлять на leaderboard
Output(data = training.getPredictData(convert.getIDs()), resultFileName = resultFile, columns = columns).output()