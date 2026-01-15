import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def read_data(filename, index=0):
    """
    Загружает данные из текстового файла.
    
    Параметры:
    filename (str): имя файла с данными
    index (int): номер колонки с датами (по умолчанию 0)
    
    Возвращает:
    pandas.Series: временной ряд с датами в качестве индекса
    """
    # Чтение данных из файла
    df = pd.read_csv(filename, sep='\s+', header=None, engine='python')
    
    print(f"Загружено {len(df)} строк, {len(df.columns)} столбцов")
    print(f"Первые 5 строк:\n{df.head()}")
    print(f"Типы данных:\n{df.dtypes}")
    
    # Пробуем разные варианты обработки дат
    try:
        # Вариант 1: если даты представлены как целые числа (номера дней с начала)
        if df[index].dtype in [np.int64, np.float64]:
            print(f"Столбец {index} содержит числовые данные")
            # Создаем даты начиная с базовой даты
            base_date = pd.Timestamp('2000-01-01')
            dates = base_date + pd.to_timedelta(df[index], unit='D')
        
        # Вариант 2: если даты в формате YYYY.MM
        elif df[index].astype(str).str.contains('\.').any():
            try:
                dates = pd.to_datetime(df[index], format='%Y.%m')
            except:
                dates = pd.to_datetime(df[index], format='%Y.%f')
        
        # Вариант 3: если даты в формате YYYY-MM-DD
        elif df[index].astype(str).str.contains('-').any():
            dates = pd.to_datetime(df[index])
        
        else:
            # Вариант 4: просто создаем индекс дат
            print("Использую автоматический индекс дат")
            dates = pd.date_range('2000-01-01', periods=len(df), freq='D')
    
    except Exception as e:
        print(f"Ошибка при обработке дат: {e}")
        # Создаем простой числовой индекс
        dates = pd.RangeIndex(start=0, stop=len(df))
    
    # Определяем столбец со значениями
    if len(df.columns) > 1:
        # Берем следующий после индекса столбец
        value_col = 1 if index == 0 else 0
        values = df[value_col].values
    else:
        # Если только один столбец, используем его как значения
        values = df[0].values
    
    # Создаем временной ряд
    time_series = pd.Series(values, index=dates)
    
    # Сортируем по дате
    time_series = time_series.sort_index()
    
    print(f"Создан временной ряд с {len(time_series)} точками")
    print(f"Диапазон дат: от {time_series.index[0]} до {time_series.index[-1]}")
    
    return time_series


# Загрузка входных данных
index = 2
print(f"Пробуем загрузить данные из data_2D.txt с индексом {index}")
data = read_data('data_2D.txt', index)

# Проверяем, что данные загружены
if not data.empty:
    print("\nПервые 5 значений временного ряда:")
    print(data.head())
    
    # Показываем информацию о диапазоне данных
    print(f"\nМинимальная дата в данных: {data.index.min()}")
    print(f"Максимальная дата в данных: {data.index.max()}")
    
    # Создаем фигуру с двумя подграфиками (один под другим)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Первый график: данные с гранулярностью на уровне года
    start = '2003'
    end = '2011'
    print(f"\nПопытка отобразить данные с {start} по {end}")
    
    # Проверяем, есть ли данные в этом диапазоне
    data_slice = data[start:end]
    print(f"Найдено {len(data_slice)} точек в диапазоне {start}:{end}")
    
    if len(data_slice) > 0:
        data_slice.plot(ax=ax1)
        ax1.set_title(f'Входные данные с {start} по {end} ({len(data_slice)} точек)')
    else:
        # Показываем сообщение, если нет данных
        ax1.text(0.5, 0.5, f'Нет данных в диапазоне {start} - {end}', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax1.transAxes, fontsize=12)
        ax1.set_title(f'Входные данные с {start} по {end} (нет данных)')
    ax1.grid(True)
    
    # Второй график: данные с гранулярностью на уровне месяца
    start = '1998-2'
    end = '2006-7'
    print(f"\nПопытка отобразить данные с {start} по {end}")
    
    data_slice = data[start:end]
    print(f"Найдено {len(data_slice)} точек в диапазоне {start}:{end}")
    
    if len(data_slice) > 0:
        data_slice.plot(ax=ax2)
        ax2.set_title(f'Входные данные с {start} по {end} ({len(data_slice)} точек)')
    else:
        ax2.text(0.5, 0.5, f'Нет данных в диапазоне {start} - {end}', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax2.transAxes, fontsize=12)
        ax2.set_title(f'Входные данные с {start} по {end} (нет данных)')
    ax2.grid(True)
    
    # Автоматическая подгонка layout'а
    plt.tight_layout()
    
    # Показываем оба графика в одном окне
    plt.show()
else:
    print("Данные не загружены или файл пуст")
