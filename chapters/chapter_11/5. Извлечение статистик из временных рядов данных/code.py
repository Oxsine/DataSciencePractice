import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd

# Имя входного файла
input_file = 'data_2D.txt'

# Альтернативный способ загрузки данных (если timeseries недоступен)
try:
    # Если файл в формате txt с числовыми данными
    data_array = np.loadtxt(input_file)
    # Предполагаем, что колонка 2 и 3 содержат нужные данные (индексация с 0)
    x1 = data_array[:, 1]  # Вторая колонка
    x2 = data_array[:, 2]  # Третья колонка
except:
    # Если данные в CSV или другом формате
    try:
        data_df = pd.read_csv(input_file, sep='\t', header=None)  # для табуляции
        x1 = data_df.iloc[:, 1].values
        x2 = data_df.iloc[:, 2].values
    except:
        # Создаем тестовые данные для демонстрации
        print("Файл не найден, создаю тестовые данные...")
        np.random.seed(42)
        n_points = 200
        t = np.linspace(0, 20, n_points)
        x1 = np.sin(t) + np.random.normal(0, 0.1, n_points)
        x2 = np.cos(t) + np.random.normal(0, 0.1, n_points)

# Создание фрейма данных Pandas для извлечения срезов
data = pd.DataFrame({'dim1': x1, 'dim2': x2})

# Извлечение максимального и минимального значений
print('\nMaximum values for each dimension:')
print(data.max())
print('\nMinimum values for each dimension:') 
print(data.min()) 

# Извлечение общего среднего и среднего по строкам
print('\nOverall mean:')
print(data.mean())
print('\nRow-wise mean:')
print(data.mean(1)[:12])  # первые 12 значений

# Построение графика скользящего среднего 
# с использованием окна шириной 24
data.rolling(center=False, window=24).mean().plot()
plt.title('Скользящее среднее')
plt.xlabel('Время/Индекс')
plt.ylabel('Значение')
plt.legend(['dim1', 'dim2'])
plt.grid(True, alpha=0.3)

# Извлечение коэффициентов корреляции
print('\nCorrelation coefficients: \n', data.corr())

# Построение графика скользящей корреляции 
# с использованием окна шириной 60
plt.figure()
plt.title('Скользящий коэффициент корреляции')
data['dim1'].rolling(window=60).corr(other=data['dim2']).plot()
plt.xlabel('Время/Индекс')
plt.ylabel('Коэффициент корреляции')
plt.grid(True, alpha=0.3)
plt.ylim(-1, 1)  # корреляция всегда между -1 и 1
plt.show()
