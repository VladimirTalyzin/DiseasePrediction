from csv import writer

class Output:
    __data = None
    __resultFileName = None
    __columns = None

    def __init__(self, data, resultFileName, columns):
        self.__data = data
        self.__resultFileName = resultFileName
        self.__columns = columns

    def output(self):
        with open(self.__resultFileName, "w", encoding = "utf8", newline = "") as csvFile:
            csvWriter = writer(csvFile, delimiter = ",")
            csvWriter.writerow(self.__columns)
            for row in self.__data:
                csvWriter.writerow(row)
