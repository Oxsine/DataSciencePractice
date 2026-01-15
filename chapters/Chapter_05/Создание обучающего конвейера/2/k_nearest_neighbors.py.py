import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

# Входные данные
X = np.array([[2.1, 1.3], [1.3, 3.2], [2.9, 2.5], [2.7, 5.4], [3.8, 0.9], [7.3, 2.1], [4.2, 6.5], [3.8, 3.7], [2.5, 4.1], [3.4, 1.9], [5.7, 3.5], [6.1, 4.3], [5.1, 2.2], [6.2, 1.1]])

# Тестовая точка данных
test_datapoint = [4.3, 2.7]

# Параметр k (число ближайших соседей)
k = 5

# Отображение входных данных на графике
plt.figure()
plt.title('Входные данные')
plt.scatter(X[:, 0], X[:, 1], marker='o', s=75, color='black')

# Построение модели на основе метода k ближайших соседей
knn_model = NearestNeighbors(n_neighbors=k, algorithm='ball_tree').fit(X)
distances, indices = knn_model.kneighbors([test_datapoint])

# Вывод 'k' ближайших соседей
print("\nК ближайших соседей:")
for rank, index in enumerate(indices[0][:k], start=1):
    print(f"{rank} ==> {X[index]}")

# Визуализация ближайших соседей вместе с тестовой точкой
plt.figure()
plt.title('Ближайшие соседи')
plt.scatter(X[:, 0], X[:, 1], marker='o', s=75, color='black')
plt.scatter(X[indices[0][:k], 0], X[indices[0][:k], 1], marker='o', s=250, facecolors='none', edgecolors='red')
plt.scatter(test_datapoint[0], test_datapoint[1], marker='x', s=100, color='blue')
plt.show()
