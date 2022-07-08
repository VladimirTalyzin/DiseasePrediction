import math
from csv import DictReader
from tensorflow import constant

# Переводим время HH:MM в количество минут с начала дня
def timeToMinutes(time):
    splitTime = time.split(':')
    return int(splitTime[0]) * 60 + int(splitTime[1]) * 1

# noinspection SpellCheckingInspection
# задаём правила преобразования колонок данных для обучения
# ключ - название колонки
# type - тип правила преобразования.
#   skip - не использовать данные колонки
#   quantitative_1_0 - количественное значение, может быть 0 или 1
#   quantitative - количественное значение. Должно быть задано max и min. Нормализуется до интервала [0, 1]
#   qualitative - качественное значение. Должен быть задан список значений в values
#     Все варианты распадаются на столько колонок, столько возможно значений. В каждой колонке будет 0 или 1
# equal - словарь замены значений
# time_transform - выполнить преобразование времени HH:MM в количество минут с начала дня
convertTrainingRules = \
{
    "Пол": {"type": "quantitative_1_0", "equal" : {"" : "0", "М" : "0", "Ж" : "1"}},
    "Семья": {"type": "qualitative", "values": ["в браке в настоящее время", "в разводе", "гражданский брак / проживание с партнером", "вдовец / вдова", "никогда не был(а) в браке", "раздельное проживание (официально не разведены)"]},
    "Этнос": {"type": "qualitative", "values": ["европейская", "другая азиатская (Корея, Малайзия, Таиланд, Вьетнам, Казахстан, Киргизия, Туркмения, Узбекистан, Таджикистан)", "прочее (любая иная этно-расовая группа, не представленная выше)"]},
    "Национальность": {"type": "qualitative", "values": ["Русские", "Белорусы", "Украинцы", "Мордва", "Татары", "Башкиры", "Молдаване", "Эстонцы", "Чуваши", "Армяне", "Азербайджанцы", "Лезгины", "Киргизы", "Казахи", "Таджики", "Буряты", "Удмурты", "Евреи", "Немцы", "Другие национальности"]},
    "Религия": {"type": "qualitative", "values": ["Христианство", "Ислам", "Индуизм", "Атеист / агностик", "Другое", "Нет"]},
    "Образование": {"type": "qualitative", "values": ["3 - средняя школа / закон.среднее / выше среднего", "5 - ВУЗ", "2 - начальная школа", "4 - профессиональное училище"]},
    "Профессия": {"type": "qualitative", "values": ["низкоквалифицированные работники", "дипломированные специалисты", "операторы и монтажники установок и машинного оборудования", "служащие", "работники,  занятые в сфере обслуживания, торговые работники магазинов и рынков", "представители   законодат.   органов   власти,  высокопостав. долж.лица и менеджеры", "техники и младшие специалисты", "ведение домашнего хозяйства", "квалифицированные работники сельского хозяйства и рыболовного", "ремесленники и представители других отраслей промышленности", "вооруженные силы"]},
    "Вы работаете?": {"type": "quantitative_1_0"},
    "Выход на пенсию": {"type": "quantitative_1_0"},
    "Прекращение работы по болезни": {"type": "quantitative_1_0"},
    "Сахарный диабет": {"type": "quantitative_1_0"},
    "Гепатит": {"type": "quantitative_1_0"},
    "Онкология": {"type": "quantitative_1_0"},
    "Хроническое заболевание легких": {"type": "quantitative_1_0"},
    "Бронжиальная астма": {"type": "quantitative_1_0"},
    "Туберкулез легких": {"type": "quantitative_1_0"},
    "ВИЧ/СПИД": {"type": "quantitative_1_0"},
    "Регулярный прим лекарственных средств": {"type": "quantitative_1_0"},
    "Травмы за год": {"type": "quantitative_1_0"},
    "Переломы": {"type": "quantitative_1_0"},
    "Статус Курения": {"type": "qualitative", "values": ["Курит", "Никогда не курил(а)", "Бросил(а)"], "equal" : {"Никогда не курил" : "Никогда не курил(а)"}},
    "Возраст курения": {"type": "quantitative", "min": 0, "max": 58, "equal" : {"" : 0}},
    "Сигарет в день": {"type": "quantitative", "min": 0, "max": 60, "equal": {"": 0, "180.0": 60, "80.0": 60}},
    "Пассивное курение": {"type": "quantitative_1_0"},
    "Частота пасс кур": {"type": "quantitative", "min": 0, "max": 28, "equal": {"": 0, "1-2 раза в неделю": 2, "3-6 раз в неделю": 6, "не менее 1 раза в день": 12, "2-3 раза в день": 21, "4 и более раз в день": 28}},
    "Алкоголь": {"type": "qualitative", "values": ["употребляю в настоящее время", "никогда не употреблял", "ранее употреблял"]},
    "Возраст алког": {"type": "quantitative", "min": 0, "max": 50, "equal": {"": 0, "63.0": 50, "60.0": 50}},
    "Время засыпания": {"type": "quantitative", "time_transform": True, "min": timeToMinutes("20:00:00"), "max": timeToMinutes("33:00:00"), "equal": {"00:00:00": "24:00:00", "00:01:00": "24:01:00", "00:00:30": "24:00:30", "00:05:00": "24:05:00", "00:10:00": "24:10:00", "00:15:00": "24:15:00", "00:30:00": "24:30:00", "01:00:00": "25:00:00", "01:20:00": "25:20:00", "01:30:00": "25:30:00", "02:00:00": "26:00:00", "02:30:00": "26:30:00", "03:00:00": "27:00:00", "04:00:00": "28:00:00", "05:00:00": "29:00:00", "09:00:00": "33:00:00", "12:00:00": "33:00:00"}},
    "Время пробуждения": {"type": "quantitative", "time_transform": True, "min": timeToMinutes("00:00:00"), "max": timeToMinutes("12:00:00")},
    "Сон после обеда": {"type": "quantitative_1_0"},
    "Спорт, клубы": {"type": "quantitative_1_0"},
    "Религия, клубы": {"type": "quantitative_1_0"},
    "Раз": {"type": "qualitative", "values": ["0", "5", "6", "7"]},
    "Два": {"type": "qualitative", "values": ["1", "2", "3", "4"]},
    "Четыре": {"type": "qualitative", "values": ["01", "02", "03", "04"]}
}

# noinspection SpellCheckingInspection
# задаём правила преобразования данных о заболеваниях
# правила аналогичны правилам преобразования данных колонок, но имеет смысл лишь тип quantitative_1_0
convertCheckRules = \
{
    "Артериальная гипертензия": {"type": "quantitative_1_0"},
    "ОНМК": {"type": "quantitative_1_0"},
    "Стенокардия, ИБС, инфаркт миокарда": {"type": "quantitative_1_0"},
    "Сердечная недостаточность": {"type": "quantitative_1_0"},
    "Прочие заболевания сердца": {"type": "quantitative_1_0"},
}

# класс чтения и конвертации данных их CSV-файлов
class Convert:
    __trainingData = {}
    __checkData = {}
    __ids = []


    # filename - CSV-файл для чтения
    # withCheckData - считать также колонки с данными заболеваний. Имеет смысл для файла со значениями для обучения
    def __init__(self, filename, withCheckData):
        # получить все колонки и все возможные значения в них из csv-файла
        self.__ids, columns = Convert.__prepareColumns__(filename, features = None, idMode = True)

        for column in columns:
            # для каждой колонки получаем её имя без начальных и конечных пробелов
            field = column["field"].strip()

            if field in convertTrainingRules:
                # если имя колонки есть в правилах преобразования колонок, то преобразовать согласно соответствующему правилу
                Convert.__transform__(self.__trainingData, field, column["values"], convertTrainingRules[field])
            elif field in convertCheckRules:
                if withCheckData:
                    # если имя колонки есть в правилах преобразования данных о заболеваниях, то преобразовать согласно соответствующему правилу
                    Convert.__transform__(self.__checkData, field, column["values"], convertCheckRules[field])
            # для поля ID никакого правила не нужно. Для остальных, если не было правила, то вывести ошибку
            elif field != "ID":
                print("UNKNOWN FIELD: " + field)

    # получить список всех ID из файла обучения, чтобы затем добавить их в файл результата
    def getIDs(self):
        return self.__ids


    # получить данные для обучения в виде тензора Numpy
    def getTrainingData(self):
        return Convert.dictionaryToTensor(self.__trainingData)


    # получить данные о заболеваниях в виде тензора Numpy
    def getCheckData(self):
        return Convert.dictionaryToTensor(self.__checkData)


    # преобразование колонки с данными, согласно правилу
    # fieldsArray - словарь, куда будут добавлены преобразованные данные
    # field - имя колонки
    # values - колонка с данными (список)
    # rule - правило преобразования
    @staticmethod
    def __transform__(fieldsArray, field, values, rule):
        # для качественных значений создаём столько колонок, сколько значений возможно
        if rule["type"] == "qualitative":
            for qualitativeField in rule["values"]:
                fieldsArray[field + "_" + qualitativeField] = {}
        else:
            # для всех других значений создаём одну колонку
            fieldsArray[field] = {}

        for idRow, value in values.items():
            # если были заданы правила замены значений, то заменяем
            if "equal" in rule and value in rule["equal"]:
                value = rule["equal"][value]

            # если было задано преобразование времени в минуты, то преобразуем
            if "time_transform" in rule and rule["time_transform"]:
                value = timeToMinutes(value)

            match rule["type"]:
                # если тип правила skip, то не добавляем данные
                case "skip":
                    pass

                # если тип правила quantitative_1_0, то получаем и записываем 0 или 1. Если в данных были иные значения, чем 0 или 1, то выводим ошибку
                case "quantitative_1_0":
                    if value == "0" or value == 0:
                        fieldsArray[field][idRow] = 0
                    elif value == "1" or value == 1:
                        fieldsArray[field][idRow] = 1
                    else:
                        print("INVALID 1,0 VALUE: " + str(value) + " FIELD: " + field)

                # если тип правила quantitative, то проверяем нахождение значения в пределах [min, max]
                # затем нормализуем значение
                case "quantitative":
                    floatValue = float(value)

                    if floatValue < rule["min"]:
                        print("VALUE LESS THEN MIN: " + str(value) + " MIN: " + str(rule["min"]) + " FIELD: " + field)
                    elif floatValue > rule["max"]:
                        print("VALUE MORE THEN MAX: " + str(value) + " MAX: " + str(rule["max"]) + " FIELD: " + field)
                    else:
                        fieldsArray[field][idRow] = (floatValue - rule["min"]) / (rule["max"] - rule["min"])

                # если тип правила qualitative, то заносим 1 в ту колонку, которая соответствует исходному значению. В остальные колонки пишем 0
                case "qualitative":
                    for qualitativeField in rule["values"]:
                        fieldsArray[field + "_" + qualitativeField][idRow] = 1 if value == qualitativeField else 0

                # для неподдерживаемых правил выводим ошибку
                case _:
                    print("UNKNOWN RULE: " + rule["type"])


    # подготовить данные csv-файла, разнеся колонки на словари по названию колонки и значениям
    # а также собрать список ID
    # filename - csv-файл
    # features - словарь с функциями генерации дополнительных колонок с данными, на основе имеющихся
    # idMode - False - распределить данные по колонкам как list, True - как словарь с ключом, равным ID
    @staticmethod
    def __prepareColumns__(filename, features, idMode):
        columns = []
        data = []
        ids = []

        # прочитать файл стандартной библиотекой Python
        with open(filename, "r", encoding = "utf8") as csvFile:
            reader = DictReader(csvFile)
            # пройти по первой строке, в которой записаны заголовки колонок
            for fieldName in reader.fieldnames:
                # для всех колонок, кроме ID_y, создать запись в результирующем словаре
                if fieldName != "ID_y":
                    columns.append({"field": fieldName, "values": {} if idMode else []})


            # прочитать все остальные строки в data
            for row in reader:
                # если заданы дополнительные колонки, создать новые данные в строке
                if features is not None:
                    for feature, function in features.items():
                        row[feature] = function(row)
                data.append(row)

        # для всех колонок, распределить собранные данные в дата по этим колонкам
        for column in columns:
            field = column["field"]

            for row in data:
                value = row[field]

                if idMode:
                    if field != "ID":
                        column["values"][row["ID"]] = value
                elif not value in column["values"]:
                    column["values"].append(value)

                # если имя колонки "ID", то добавить значение в список ids
                if field == "ID":
                    ids.append(value)

        return ids, columns


    # преобразовать собранные в словарь и подготовленные данные в numpy-тензор
    @staticmethod
    def dictionaryToTensor(dictionary):

        # собрать все данные в двумерный словарь
        # первое измерение - id строки
        # второе измерение - название колонки
        collect = {}
        for field, values in dictionary.items():
            for idRow, value in values.items():
                if not idRow in collect:
                    collect[idRow] = {}

                collect[idRow][field] = value

        # собрать двумерный словарь в двумерный массив
        result = []
        for _, values in collect.items():
            row = []
            for value in values.values():
                # если число не имеет дробной части, записываем его как тип int, иначе как тип float
                if float(int(value)) != float(value):
                    parsedValue = float(value)
                else:
                    parsedValue = int(value)

                if math.isnan(parsedValue):
                    print("VALUE IS NAN!")
                else:
                    row.append(parsedValue)

            result.append(row)

        # преобразовать собранный двумерный массив в numpy-тензор
        return constant(result).numpy()