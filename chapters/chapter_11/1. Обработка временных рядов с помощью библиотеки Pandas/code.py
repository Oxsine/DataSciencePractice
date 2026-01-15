import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Создаем тестовые данные, если файл не существует
def create_test_data(filename='data_2D.txt'):
    # Генерируем данные за 2 года (24 месяца)
    years = []
    months = []
    data1 = []
    data2 = []
    
    for year in [2020, 2021]:
        for month in range(1, 13):
            years.append(year)
            months.append(month)
            # Генерируем случайные данные с трендом
            data1.append(100 + month * 2 + np.random.randn() * 5)
            data2.append(200 - month * 1.5 + np.random.randn() * 3)
    
    # Сохраняем в файл
    with open(filename, 'w') as f:
        for i in range(len(years)):
            f.write(f"{years[i]} {months[i]} {data1[i]:.2f} {data2[i]:.2f}\n")
    print(f"Создан тестовый файл: {filename}")

def read_data(input_file, index):
    # Чтение данных из входного файла
    input_data = np.loadtxt(input_file, delimiter=None)
                             
    # Лямбда-функция для преобразования 
    # строк в формат данных Pandas
    to_date = lambda x, y: str(int(x)) + '-' + str(int(y))

    # Извлечение начальной даты
    start = to_date(input_data[0, 0], input_data[0, 1])

    # Извлечение конечной даты
    if input_data[-1, 1] == 12:
        year = input_data[-1, 0] + 1
        month = 1
    else:
        year = input_data[-1, 0]
        month = input_data[-1, 1] + 1
    
    end = to_date(year, month)

    # Создание списка дат с ежемесячной частотой
    date_indices = pd.date_range(start, end, freq="M", inclusive='left')

    # Добавление меток во входные данные для создания
    # временного ряда данных
    output = pd.Series(input_data[:, index], index=date_indices)
    return output

if __name__ == '__main__':
    # Имя входного файла
    input_file = 'data_2D.txt'
    
    # Создаем тестовые данные, если файл не существует
    try:
        with open(input_file, 'r'):
            pass
    except FileNotFoundError:
        print("Файл не найден. Создаю тестовые данные...")
        create_test_data(input_file)

    # Указание столбцов, подлежащих преобразованию 
    # во временной ряд данных
    indices = [2, 3]

    # Итерирование по столбцам и построение графика данных
    for index in indices:
        # Преобразование столбца в формат временного ряда
        timeseries = read_data(input_file, index)

        # Построение графика
        plt.figure () 
        timeseries.plot () 
        plt.title ("Размерность " + str (index - 1)) 
        plt.show() 
