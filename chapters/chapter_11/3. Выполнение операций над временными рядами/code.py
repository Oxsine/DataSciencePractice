import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

# Определяем функцию read_data
def read_data(filename, index=0):
    """
    Загружает данные из текстового файла.
    
    Параметры:
    filename (str): имя файла с данными
    index (int): индекс колонки со значениями (даты будут созданы автоматически)
    
    Возвращает:
    pandas.Series: временной ряд с датами в качестве индекса
    """
    try:
        # Чтение данных из файла
        df = pd.read_csv(filename, sep='\s+', header=None, engine='python')
        
        print(f"Загружено {len(df)} строк, {len(df.columns)} столбцов")
        print(f"Первые 5 строк:\n{df.head()}")
        
        # Если index указывает на колонку со значениями, а не с датами
        # Создаем числовой индекс
        values = df[index].values
        
        # Создаем даты начиная с 1960-01-01 (предполагаемый диапазон)
        dates = pd.date_range('1960-01-01', periods=len(values), freq='D')
        
        # Создаем временной ряд
        time_series = pd.Series(values, index=dates)
        
        print(f"Создан временной ряд с {len(time_series)} точками")
        print(f"Диапазон дат: от {time_series.index[0]} до {time_series.index[-1]}")
        
        return time_series
        
    except Exception as e:
        print(f"Ошибка при чтении файла {filename}: {e}")
        return pd.Series()

# Имя входного файла
input_file = 'data_2D.txt'

print("=" * 50)
print("Загрузка первого временного ряда (индекс 2)...")
X1 = read_data(input_file, 2)

print("\n" + "=" * 50)
print("Загрузка второго временного ряда (индекс 3)...")
x2 = read_data(input_file, 3)

# Проверяем, что оба ряда загружены
if X1.empty or x2.empty:
    print("\nОшибка: один или оба временных ряда пусты!")
    exit()

# Выравниваем индексы (берем минимальную длину)
min_length = min(len(X1), len(x2))
print(f"\nМинимальная длина: {min_length}")

# Берем первые min_length элементов из каждого ряда
X1_aligned = X1.iloc[:min_length]
x2_aligned = x2.iloc[:min_length]

# Создание фрейма данных Pandas
print("\n" + "=" * 50)
print("Создание DataFrame...")
data = pd.DataFrame({'dim1': X1_aligned.values, 'dim2': x2_aligned.values}, 
                    index=X1_aligned.index)

print(f"\nСоздан DataFrame с {len(data)} строками")
print("Первые 5 строк:")
print(data.head())
print("\nИнформация о DataFrame:")
print(data.info())

# Показываем информацию о диапазоне данных
print(f"\nМинимальная дата: {data.index.min()}")
print(f"Максимальная дата: {data.index.max()}")
print(f"Диапазон значений dim1: от {data['dim1'].min():.2f} до {data['dim1'].max():.2f}")
print(f"Диапазон значений dim2: от {data['dim2'].min():.2f} до {data['dim2'].max():.2f}")

# Создаем фигуру с несколькими графиками
fig, axes = plt.subplots(3, 1, figsize=(12, 12))

# 1. Построение графика с 1968 по 1975
start = '1968'
end = '1975'
print(f"\n" + "=" * 50)
print(f"Попытка отобразить данные с {start} по {end}")

if len(data[start:end]) > 0:
    data[start:end].plot(ax=axes[0])
    axes[0].set_title(f'Наложение двух графиков ({start} - {end})')
    axes[0].grid(True)
    print(f"Найдено {len(data[start:end])} точек в диапазоне")
else:
    # Если нет данных в указанном диапазоне, показываем все данные
    data.plot(ax=axes[0])
    axes[0].set_title('Наложение двух графиков (весь диапазон)')
    axes[0].grid(True)
    print(f"Нет данных в диапазоне {start} - {end}, показываю весь диапазон")

# 2. Фильтрация с использованием условий
print(f"\n" + "=" * 50)
print("Фильтрация данных...")
filtered_data = data[(data['dim1'] < 45) & (data['dim2'] > 30)]
print(f"Найдено {len(filtered_data)} строк после фильтрации (dim1 < 45 и dim2 > 30)")

if len(filtered_data) > 0:
    filtered_data.plot(ax=axes[1])
    axes[1].set_title('dim1 < 45 и dim2 > 30')
    axes[1].grid(True)
else:
    axes[1].text(0.5, 0.5, 'Нет данных, удовлетворяющих условию фильтрации',
                horizontalalignment='center', verticalalignment='center',
                transform=axes[1].transAxes, fontsize=12)
    axes[1].set_title('dim1 < 45 и dim2 > 30 (нет данных)')
    axes[1].grid(True)

# 3. Сложение двух фреймов данных
print(f"\n" + "=" * 50)
print("Сложение двух временных рядов...")

if len(data[start:end]) > 0:
    diff = data[start:end]['dim1'] + data[start:end]['dim2']
    diff.plot(ax=axes[2], color='red')
    axes[2].set_title(f'Сумма (dim1 + dim2) для периода {start} - {end}')
    print(f"Вычислена сумма для {len(diff)} точек")
else:
    # Если нет данных в указанном диапазоне, складываем все данные
    diff = data['dim1'] + data['dim2']
    diff.plot(ax=axes[2], color='red')
    axes[2].set_title('Сумма (dim1 + dim2) для всего диапазона')
    print(f"Вычислена сумма для всего диапазона ({len(diff)} точек)")

axes[2].grid(True)

plt.tight_layout()
plt.show()

# Дополнительная статистика
print(f"\n" + "=" * 50)
print("СТАТИСТИКА:")
print(f"Всего точек: {len(data)}")
print(f"Среднее dim1: {data['dim1'].mean():.2f}")
print(f"Среднее dim2: {data['dim2'].mean():.2f}")
print(f"Стандартное отклонение dim1: {data['dim1'].std():.2f}")
print(f"Стандартное отклонение dim2: {data['dim2'].std():.2f}")
print(f"Корреляция между dim1 и dim2: {data['dim1'].corr(data['dim2']):.3f}")
