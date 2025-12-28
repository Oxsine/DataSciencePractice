import datetime
import json
import numpy as np
import yfinance as yf
from sklearn import covariance, cluster

# Входной файл с символическими обозначениями компаний
input_file = '../data/company_symbol_mapping.json'

# Загрузка привязок символов компаний к их полным названиям
with open(input_file, 'r') as f:
    company_symbols_map = json.load(f)

# Преобразуем в массивы, но будем фильтровать
all_symbols, all_names = list(company_symbols_map.keys()), list(company_symbols_map.values())

# Задаём даты
start_date = datetime.datetime(2003, 7, 3)
end_date = datetime.datetime(2007, 5, 4)

# Загрузка котировок для каждого символа с обработкой ошибок
valid_quotes = []
valid_symbols = []
valid_names = []
failed_symbols = []

print("Загрузка данных...")
for symbol, name in zip(all_symbols, all_names):
    try:
        # Пытаемся загрузить данные
        data = yf.download(symbol, start=start_date, end=end_date, progress=False)
        
        # Проверяем, что данные не пустые и содержат необходимые колонки
        if not data.empty and 'Open' in data.columns and 'Close' in data.columns:
            if len(data) > 50:  # Убедимся, что есть достаточное количество точек данных
                valid_quotes.append(data)
                valid_symbols.append(symbol)
                valid_names.append(name)
                print(f"✓ {symbol} ({name}): {len(data)} дней")
            else:
                failed_symbols.append(symbol)
                print(f"✗ {symbol}: недостаточно данных ({len(data)} дней)")
        else:
            failed_symbols.append(symbol)
            print(f"✗ {symbol}: пустые данные")
            
    except Exception as e:
        failed_symbols.append(symbol)
        print(f"✗ {symbol}: ошибка - {str(e)[:50]}...")

# Проверяем, есть ли достаточное количество валидных данных
if len(valid_quotes) < 5:
    print(f"\nНедостаточно данных для анализа. Успешно загружено: {len(valid_quotes)} компаний")
    print(f"Не удалось загрузить: {failed_symbols}")
    exit()

print(f"\nВсего успешно загружено: {len(valid_quotes)} компаний")
print(f"Не удалось загрузить: {len(failed_symbols)} компаний")

# Находим общие даты для всех акций
# Преобразуем индексы дат в строки для сравнения
date_strings = []
for q in valid_quotes:
    date_strings.append(set(q.index.strftime('%Y-%m-%d')))

# Находим пересечение всех дат
common_dates = set.intersection(*date_strings)
common_dates = sorted(list(common_dates))

print(f"\nОбщее количество торговых дней: {len(common_dates)}")

if len(common_dates) < 50:
    print(f"Слишком мало общих торговых дней: {len(common_dates)}")
    exit()

# Фильтруем данные по общим датам и извлекаем цены открытия и закрытия
opening_quotes_list = []
closing_quotes_list = []

print("\nПодготовка данных...")
for q in valid_quotes:
    # Создаем строковое представление дат для фильтрации
    q_dates = q.index.strftime('%Y-%m-%d')
    mask = [date in common_dates for date in q_dates]
    
    filtered_data = q[mask]
    
    # Сортируем по дате
    filtered_data = filtered_data.sort_index()
    
    # Извлекаем данные и преобразуем в 1D массивы
    opening_quotes_list.append(filtered_data['Open'].values.flatten())
    closing_quotes_list.append(filtered_data['Close'].values.flatten())

# Преобразуем в numpy массивы
opening_quotes = np.vstack(opening_quotes_list)
closing_quotes = np.vstack(closing_quotes_list)

print(f"\nРазмерность opening_quotes: {opening_quotes.shape}")
print(f"Размерность closing_quotes: {closing_quotes.shape}")

# Вычисление разности котировок
quotes_diff = closing_quotes - opening_quotes

# Нормализация данных
X = quotes_diff.T  # Транспонируем для получения (дни × компании)

print(f"\nРазмерность X до очистки: {X.shape}")

# Удаляем столбцы с нулевой дисперсией или NaN
valid_indices = []
for i in range(X.shape[1]):
    col = X[:, i]
    col_std = np.std(col)
    col_has_nan = np.any(np.isnan(col))
    if col_std > 0.0001 and not col_has_nan:
        valid_indices.append(i)
    else:
        print(f"Удаляем {valid_names[i]} из-за нулевой дисперсии или NaN")

X_clean = X[:, valid_indices]
valid_names_clean = [valid_names[i] for i in valid_indices]

if X_clean.shape[1] < 5:
    print(f"\nНедостаточно данных для анализа (нужно минимум 5 валидных акций, доступно: {X_clean.shape[1]})")
    exit()

print(f"\nОсталось компаний для анализа: {X_clean.shape[1]}")

# Стандартизируем данные
X_mean = np.mean(X_clean, axis=0)
X_std = np.std(X_clean, axis=0)
X_std[X_std == 0] = 1  # Защита от деления на 0
X_standardized = (X_clean - X_mean) / X_std

print(f"\nНачинаем анализ с {X_standardized.shape[1]} компаниями...")

try:
    # Сначала вычисляем корреляционную матрицу
    correlation_matrix = np.corrcoef(X_standardized.T)
    
    # Проверяем корреляционную матрицу на NaN
    if np.any(np.isnan(correlation_matrix)):
        print("Корреляционная матрица содержит NaN значения. Пытаемся исправить...")
        # Заменяем NaN на 0
        correlation_matrix = np.nan_to_num(correlation_matrix)
    
    print(f"Размер корреляционной матрицы: {correlation_matrix.shape}")
    
    # Обучение модели ковариационной связи
    # Используем GraphicalLassoCV вместо GraphLassoCV
    edge_model = covariance.GraphicalLassoCV()
    
    with np.errstate(invalid='ignore', divide='ignore'):
        edge_model.fit(X_standardized)
    
    # Проведение кластеризации с использованием affinity propagation
    # Affinity propagation требует неотрицательной матрицы сходства
    similarity = correlation_matrix
    
    # Проверяем, что матрица не слишком велика для affinity propagation
    if len(valid_names_clean) > 100:
        print("Слишком много компаний для affinity propagation. Используем агреломеративную кластеризацию.")
        from sklearn.cluster import AgglomerativeClustering
        clustering = AgglomerativeClustering(n_clusters=min(10, len(valid_names_clean)//5), 
                                           affinity='precomputed', 
                                           linkage='average')
        labels = clustering.fit_predict(1 - similarity)
    else:
        # Используем DBSCAN как более стабильную альтернативу
        from sklearn.cluster import DBSCAN
        clustering = DBSCAN(eps=0.5, min_samples=2, metric='precomputed')
        labels = clustering.fit_predict(1 - similarity)
    
    num_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    
    if num_clusters == 0:
        print("Не удалось обнаружить кластеры. Все компании в одном кластере.")
        labels = np.zeros(len(valid_names_clean), dtype=int)
        num_clusters = 1
    
    # Вывод результатов
    print(f"\n=== РЕЗУЛЬТАТЫ КЛАСТЕРИЗАЦИИ ===")
    print(f"Найдено кластеров: {num_clusters}")
    
    for i in range(num_clusters):
        cluster_indices = np.where(labels == i)[0]
        cluster_companies = [valid_names_clean[idx] for idx in cluster_indices]
        print(f"\nКластер {i + 1} ({len(cluster_companies)} компаний):")
        for company in cluster_companies:
            print(f"  • {company}")
    
    # Вывод выбросов (если есть)
    outlier_indices = np.where(labels == -1)[0]
    if len(outlier_indices) > 0:
        outlier_companies = [valid_names_clean[idx] for idx in outlier_indices]
        print(f"\nВыбросы ({len(outlier_companies)} компаний):")
        for company in outlier_companies:
            print(f"  • {company}")
    
    # Визуализация корреляционной матрицы
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 10))
    plt.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Корреляция')
    plt.title('Корреляционная матрица акций')
    plt.xlabel('Компании')
    plt.ylabel('Компании')
    
    # Сохраняем график
    plt.savefig('correlation_matrix.png', dpi=150, bbox_inches='tight')
    print("\nГрафик корреляционной матрицы сохранен как 'correlation_matrix.png'")
    
except Exception as e:
    print(f"\nОшибка при обучении модели или кластеризации: {e}")
    import traceback
    traceback.print_exc()
    
    # Альтернативный подход: простая кластеризация по секторам
    print("\nПробуем альтернативный подход...")
    
    # Вычисляем корреляционную матрицу, если она еще не вычислена
    if 'correlation_matrix' not in locals():
        correlation_matrix = np.corrcoef(X_standardized.T)
    
    # Создаем простые кластеры на основе корреляций
    from scipy.cluster.hierarchy import dendrogram, linkage
    from scipy.spatial.distance import squareform
    
    # Преобразуем корреляцию в расстояние
    distance_matrix = 1 - correlation_matrix
    np.fill_diagonal(distance_matrix, 0)
    
    # Иерархическая кластеризация
    Z = linkage(squareform(distance_matrix), method='average')
    
    # Визуализация дендрограммы
    plt.figure(figsize=(12, 8))
    dendrogram(Z, labels=valid_names_clean, leaf_rotation=90)
    plt.title('Дендрограмма корреляций акций')
    plt.xlabel('Компании')
    plt.ylabel('Расстояние')
    plt.tight_layout()
    plt.savefig('dendrogram.png', dpi=150, bbox_inches='tight')
    print("Дендрограмма сохранена как 'dendrogram.png'")
    
    # Формируем кластеры на основе дендрограммы
    from scipy.cluster.hierarchy import fcluster
    labels = fcluster(Z, t=0.7, criterion='distance')
    num_clusters = len(set(labels))
    
    print(f"\n=== РЕЗУЛЬТАТЫ ИЕРАРХИЧЕСКОЙ КЛАСТЕРИЗАЦИИ ===")
    print(f"Найдено кластеров: {num_clusters}")
    
    for i in range(1, num_clusters + 1):
        cluster_indices = np.where(labels == i)[0]
        cluster_companies = [valid_names_clean[idx] for idx in cluster_indices]
        print(f"\nКластер {i} ({len(cluster_companies)} компаний):")
        for company in cluster_companies:
            print(f"  • {company}")

print(f"\nАнализ завершен!")