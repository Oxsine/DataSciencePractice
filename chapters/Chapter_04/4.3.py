import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

# Загрузка данных из входного файла
X = np.loadtxt('data_quality.txt',  delimiter=',')

# Инициализация переменных
scores = [ ]
values = np.arange(2, 10)

# Итерирование в определенном диапазоне значений
for num_clusters in values:
    #  Обучение модели кластеризации КМеаns
    kmeans = KMeans(init='k-means++', n_clusters = num_clusters, n_init = 10)
    kmeans.fit(X)

# Получение силуэтной оценки
score = metrics.silhouette_score(X, kmeans.labels_,
metric='euclidean', sample_size=len(X))

# Вывод силуэтной оценки
print("\nNumber of clusters =", num_clusters)
print("Silhouette score =", score)
scores.append(score)

# Отображение силуэтных оценок на графике
plt.figure ()
plt.bar(values, scores, width=0.7, color='black', align='center')
plt.title('Зависимость силуэтной оценки от количества кластеров')

# Извлечение наилучшей оценки и оптимального количества кластеров
Num_clusters = np.argmax(scores) + values[0]
print("\nOptimal number of clusters =" , num_clusters)

# Отображение данных на графике
plt.figure ()
plt.scatter(X[: ,0], X[: ,1 ] , color='black', s=80, marker='o', facecolors='none')
x_min, x_max = X[:, 0] .min () - 1, X[:, 0] .max( ) + 1
y_min, y_max = X[:, 1] .min () - 1, X[:, 1] .max() + 1
plt.title('Входные данные')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(( ))
plt.yticks(( ))
plt.show( )
