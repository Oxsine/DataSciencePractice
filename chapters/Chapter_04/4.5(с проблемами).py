import datetime
import json
import numpy as np
import yfinance as yf
from sklearn import covariance, cluster

# Входной файл с символическими обозначениями компаний
input_file = 'company_symbol_mapping.json'

# Загрузка привязок символов компаний к их полным названиям
with open(input_file, 'r') as f:
    company_symbols_map = json.load(f)

symbols, names = np.array(list(company_symbols_map.items())).T

# Задаём дат
start_date = datetime.datetime(2003, 7, 3)
end_date = datetime.datetime(2007, 5, 4)

# Загрузка котировок для каждого символа
quotes = []
for symbol in symbols:
    data = yf.download(symbol, start=start_date, end=end_date)
    quotes.append(data)

# Извлечение данных открытия и закрытия
opening_quotes = np.array([q['Open'].values for q in quotes])
closing_quotes = np.array([q['Close'].values for q in quotes])

# Проверка и корректировка длины данных, если есть пропуски
min_length = min(q.shape[0] for q in quotes)
opening_quotes = opening_quotes[:, :min_length]
closing_quotes = closing_quotes[:, :min_length]

# Вычисление разности котировок
quotes_diff = closing_quotes - opening_quotes

# Нормализация данных
X = quotes_diff.copy().T
X /= X.std(axis=0)

# Обучение модели ковариационной связи
edge_model = covariance.GraphLassoCV()

with np.errstate(invalid='ignore'):
    edge_model.fit(X)

# Получение ковариационной матрицы
covariance_matrix = edge_model.covariance_

# Построение матрицы сходства
correlation_matrix = np.corrcoef(X.T)

# Расчет сходства для кластеризации
similarity = correlation_matrix

# Проведение кластеризации с использованием affinity propagation
labels = cluster.affinity_propagation(similarity, affinity='precomputed')
num_clusters = len(np.unique(labels))

# Вывод результатов
for i in range(num_clusters):
    # Отбор названий компаний в данном кластере
    print("Cluster", i + 1, "==>", ', '.join(names[labels == i]))
