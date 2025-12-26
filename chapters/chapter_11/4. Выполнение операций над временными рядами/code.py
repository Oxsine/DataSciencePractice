import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt 

# Имя входного файла
input_file = 'data_2D.txt'

def load_or_create_data(input_file):
    """Загружает данные из файла или создает тестовые данные"""
    try:
        # Пробуем загрузить данные
        data_array = np.loadtxt(input_file)
        
        # Проверяем размерность
        print(f"Размер загруженных данных: {data_array.shape}")
        
        if len(data_array.shape) == 1:
            raise ValueError("Загружен 1D массив, требуется 2D")
        
        # Берем нужные столбцы
        if data_array.shape[1] >= 3:
            X1 = data_array[:, 1]  # Второй столбец
            X2 = data_array[:, 2]  # Третий столбец
        elif data_array.shape[1] == 2:
            X1 = data_array[:, 0]  # Первый столбец
            X2 = data_array[:, 1]  # Второй столбец
        else:
            raise ValueError(f"Неожиданное количество столбцов: {data_array.shape[1]}")
            
        return X1, X2, data_array.shape[0]
        
    except Exception as e:
        print(f"Ошибка загрузки: {e}")
        print("Создание тестовых данных...")
        
        # Создаем тестовые данные
        n_points = 200
        t = np.linspace(0, 4*np.pi, n_points)
        X1 = 45 + 10 * np.sin(t) + np.random.normal(0, 3, n_points)
        X2 = 35 + 8 * np.cos(t) + np.random.normal(0, 2, n_points)
        
        # Сохраняем тестовые данные
        test_data = np.column_stack([range(n_points), X1, X2])
        np.savetxt(input_file, test_data, fmt='%.6f')
        print(f"Создан тестовый файл {input_file} с {n_points} точками")
        
        return X1, X2, n_points

def create_dataframe(X1, X2, n_points):
    """Создает DataFrame с датами в качестве индекса"""
    # Создаем даты для индекса (месячные данные)
    dates = pd.date_range(start='1960-01-01', periods=n_points, freq='ME')
    
    data = pd.DataFrame({'dim1': X1, 'dim2': X2}, index=dates)
    
    print(f"\nСоздан DataFrame с {len(data)} записями")
    print(f"Диапазон дат: от {data.index[0].date()} до {data.index[-1].date()}")
    print(f"\nПервые 5 строк данных:")
    print(data.head())
    print(f"\nОсновная статистика:")
    print(data.describe())
    
    return data

def plot_date_range(data, start_date, end_date, title_suffix=''):
    """Строит график для заданного диапазона дат"""
    try:
        # Преобразуем строки в Timestamp для сравнения
        start_ts = pd.Timestamp(start_date)
        end_ts = pd.Timestamp(end_date)
        
        # Проверяем, что диапазон валиден
        if start_ts > end_ts:
            print(f"Ошибка: начальная дата {start_date} позже конечной {end_date}")
            return None
            
        # Фильтруем данные по диапазону
        mask = (data.index >= start_ts) & (data.index <= end_ts)
        filtered = data.loc[mask]
        
        if len(filtered) == 0:
            print(f"Нет данных в диапазоне {start_date} - {end_date}")
            print(f"Доступный диапазон: {data.index[0].date()} - {data.index[-1].date()}")
            return None
            
        plt.figure(figsize=(12, 6))
        filtered.plot()
        plt.title(f'Наложение двух графиков {title_suffix}')
        plt.xlabel('Дата')
        plt.ylabel('Значения')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return filtered
        
    except Exception as e:
        print(f"Ошибка при построении графика: {e}")
        return None

def main():
    # Загрузка или создание данных
    X1, X2, n_points = load_or_create_data(input_file)
    
    # Создание DataFrame
    data = create_dataframe(X1, X2, n_points)
    
    # График 1: Диапазон 1968-1975
    start1, end1 = '1968-01-01', '1975-12-31'
    plot_date_range(data, start1, end1, f'({start1[:4]}-{end1[:4]})')
    
    # График 2: Фильтрация данных
    plt.figure(figsize=(12, 6))
    
    # Применяем фильтр
    filtered_mask = (data['dim1'] < 45) & (data['dim2'] > 30)
    filtered_data = data[filtered_mask]
    
    if not filtered_data.empty:
        # Подграфик 1: отфильтрованные данные
        plt.subplot(2, 1, 1)
        filtered_data.plot(ax=plt.gca())
        plt.title(f'Фильтр: dim1 < 45 и dim2 > 30 ({len(filtered_data)} записей)')
        plt.xlabel('Дата')
        plt.ylabel('Значения')
        plt.grid(True, alpha=0.3)
        
        # Подграфик 2: распределение фильтрованных значений
        plt.subplot(2, 1, 2)
        plt.hist(filtered_data['dim1'], bins=30, alpha=0.7, label='dim1 (фильтр)')
        plt.hist(filtered_data['dim2'], bins=30, alpha=0.7, label='dim2 (фильтр)', color='orange')
        plt.title('Распределение отфильтрованных значений')
        plt.xlabel('Значения')
        plt.ylabel('Частота')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        print(f"\nНайдено {len(filtered_data)} записей по фильтру")
        print(f"Процент от общего количества: {len(filtered_data)/len(data)*100:.1f}%")
    else:
        print("\nНет данных, соответствующих условиям фильтрации")
        
        # Показываем распределение для понимания
        plt.subplot(2, 1, 1)
        plt.hist(data['dim1'], bins=30, alpha=0.7, label='dim1', density=True)
        plt.axvline(x=45, color='r', linestyle='--', linewidth=2, label='Порог 45')
        plt.title('Распределение dim1')
        plt.xlabel('Значения dim1')
        plt.ylabel('Плотность')
        plt.legend()
        
        plt.subplot(2, 1, 2)
        plt.hist(data['dim2'], bins=30, alpha=0.7, label='dim2', color='orange', density=True)
        plt.axvline(x=30, color='r', linestyle='--', linewidth=2, label='Порог 30')
        plt.title('Распределение dim2')
        plt.xlabel('Значения dim2')
        plt.ylabel('Плотность')
        plt.legend()
    
    plt.tight_layout()
    
    # График 3: Сумма dim1 и dim2
    plt.figure(figsize=(12, 6))
    
    # Пробуем построить для диапазона 1968-1975
    start2, end2 = '1968-01-01', '1975-12-31'
    mask = (data.index >= pd.Timestamp(start2)) & (data.index <= pd.Timestamp(end2))
    
    if mask.any():
        period_data = data.loc[mask]
        sum_series = period_data['dim1'] + period_data['dim2']
        title_period = f'({start2[:4]}-{end2[:4]})'
    else:
        sum_series = data['dim1'] + data['dim2']
        title_period = 'за весь период'
        print(f"\nДиапазон {start2}-{end2} не найден, используем все данные")
    
    # Построение графика суммы
    plt.subplot(2, 1, 1)
    sum_series.plot(color='green', linewidth=2)
    plt.title(f'Сумма (dim1 + dim2) {title_period}')
    plt.xlabel('Дата')
    plt.ylabel('Сумма')
    plt.grid(True, alpha=0.3)
    
    # Гистограмма суммы
    plt.subplot(2, 1, 2)
    plt.hist(sum_series, bins=30, alpha=0.7, color='green', edgecolor='black')
    plt.title('Распределение суммы dim1 + dim2')
    plt.xlabel('Сумма')
    plt.ylabel('Частота')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Дополнительная статистика
    print("\n" + "="*60)
    print("ПОДРОБНАЯ СТАТИСТИКА ДАННЫХ:")
    print("="*60)
    
    print(f"\nОбщая статистика:")
    print(f"Всего записей: {len(data)}")
    print(f"Диапазон дат: {data.index[0].date()} - {data.index[-1].date()}")
    
    print(f"\ndim1:")
    print(f"  Среднее: {data['dim1'].mean():.2f}")
    print(f"  Стандартное отклонение: {data['dim1'].std():.2f}")
    print(f"  Минимум: {data['dim1'].min():.2f}")
    print(f"  Максимум: {data['dim1'].max():.2f}")
    print(f"  Медиана: {data['dim1'].median():.2f}")
    
    print(f"\ndim2:")
    print(f"  Среднее: {data['dim2'].mean():.2f}")
    print(f"  Стандартное отклонение: {data['dim2'].std():.2f}")
    print(f"  Минимум: {data['dim2'].min():.2f}")
    print(f"  Максимум: {data['dim2'].max():.2f}")
    print(f"  Медиана: {data['dim2'].median():.2f}")
    
    correlation = data['dim1'].corr(data['dim2'])
    print(f"\nКорреляция между dim1 и dim2: {correlation:.3f}")
    
    if abs(correlation) > 0.7:
        print("  Сильная корреляция")
    elif abs(correlation) > 0.3:
        print("  Умеренная корреляция")
    else:
        print("  Слабая корреляция")
    
    print(f"\nСумма (dim1 + dim2):")
    total_sum = data['dim1'] + data['dim2']
    print(f"  Среднее: {total_sum.mean():.2f}")
    print(f"  Стандартное отклонение: {total_sum.std():.2f}")
    print(f"  Минимум: {total_sum.min():.2f}")
    print(f"  Максимум: {total_sum.max():.2f}")
    
    plt.show()

if __name__ == "__main__":
    main()
