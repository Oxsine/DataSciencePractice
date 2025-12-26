import csv
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift, estimate_bandwidth

# Загрузка данных из входного файла
input_file = 'sales.csv'
with open(input_file, 'r') as file:
    file_reader = csv.reader(file, delimiter=',')
    X = []
    for count, row in enumerate(file_reader):
        if not count:
            names = row[1:]
            continue
        # Обработка строк данных, пропуская первый столбец, если там названия или индексы
        X.append([float(x) for x in row[1:11]])

# Преобразование данных в массив numpy
X = np.array(X)

# Оценка ширины окна входных данных
bandwidth = estimate_bandwidth(X, quantile=0.8, n_samples=len(X))

# Вычисление кластеризации методом сдвига среднего
meanshift_model = MeanShift(bandwidth=bandwidth, bin_seeding=True)
meanshift_model.fit(X)

#Извлечение маркеров и центров каждого кластера
labels = meanshift_model.labels_
cluster_centers = meanshift_model.cluster_centers_
num_clusters = len(np.unique(labels))

#Вывод количества кластеров и их центров
print("\nNumber of clusters in input data =", num_clusters)
print("Centers of clusters:")
print('\t'.join([f"{name[:3]}" for name in names]))
for cluster_center in cluster_centers:
    print('\t'.join([str(int(x)) for x in cluster_center]))

# Извлечение двух признаков в целях визуализации
cluster_centers_2d = cluster_centers[:, 1:3]

# Отображение центров кластеров
plt.figure()
plt.scatter(cluster_centers_2d[:, 0], cluster_centers_2d[:, 1],
            s=120, edgecolors='black', facecolors='none')

offset = 0.25
plt.xlim(cluster_centers_2d[:, 0].min() - offset * np.ptp(cluster_centers_2d[:, 0]),
         cluster_centers_2d[:, 0].max() + offset * np.ptp(cluster_centers_2d[:, 0]))
plt.ylim(cluster_centers_2d[:, 1].min() - offset * np.ptp(cluster_centers_2d[:, 1]),
         cluster_centers_2d[:, 1].max() + offset * np.ptp(cluster_centers_2d[:, 1]))

plt.title('Центры 2D-кластеров')
plt.show()
