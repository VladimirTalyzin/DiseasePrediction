from training import Training, TrainingEngine
from convert import Convert
from config import checkFile, trainFile, columns, resultFile, resultCheckTrainingFile
from output import Output

# прочитать модели из подготовленных файлов моделей
training = Training(
    # путь к файлу модели, без указания расширения, так как оно зависит от типа модели
    filename = "models/diseases",
    # тип модели. Поддерживается KERAS и SCIKIT_LEARN
    engine = TrainingEngine.KERAS,
    # выполнить автоматическое масштабирование данных в интервал [0, 1]. Не требуется, так как задано ручное масштабирование
    withScale = False,
    # режим, когда для каждой колонки берётся своя подготовленная модель
    columnsMode = True)

# подготовить данные из файла тренировки, для тестирования качества моделей. Предсказанные данные не должны
# сильно отличаться от тех, что реально указаны в файле тренировки
convert = Convert(filename = trainFile, withCheckData = True)
# выполнить предсказание
training.predict(convert.getTrainingData(), len(columns) - 1)
# записать полученное предсказание в файл resultCheckTrainingFile. По нему можно вручную проверить качество работы моделей
Output(data = training.getPredictData(convert.getIDs()), resultFileName = resultCheckTrainingFile, columns = columns).output()


# подготовить данные из файла для предсказания заболеваний
convert = Convert(filename = checkFile, withCheckData = False)
# предсказать данные по заболеваниям
training.predict(convert.getTrainingData(), len(columns) - 1)
# записать полученные предсказания в файл resultFile. Этот файл надо оправлять на leaderboard
Output(data = training.getPredictData(convert.getIDs()), resultFileName = resultFile, columns = columns).output()