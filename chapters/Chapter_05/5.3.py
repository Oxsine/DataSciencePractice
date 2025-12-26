import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn import neighbors, datasets

# Загрузка входных данных
input_file = 'data.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1].astype(int)

# Отображение входных данных на графике
plt.figure()
plt.title('Входные данные')
marker_shapes = ['o', 'v', 's', '^', 'p', '*', 'h', 'D']
mapper = [marker_shapes[i % len(marker_shapes)] for i in y]
for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], marker=mapper[i], s=75, edgecolors='black', facecolors='none')

# Количество ближайших соседей
num_neighbors = 12

# Шаг сетки визуализации
step_size = 0.01

# Создание классификатора на основе метода К ближайших соседей
classifier = neighbors.KNeighborsClassifier(n_neighbors=num_neighbors, weights='distance')

# Обучение модели
classifier.fit(X, y)

# Создание сетки для отображения границ
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_values, y_values = np.meshgrid(np.arange(x_min, x_max, step_size),
                                   np.arange(y_min, y_max, step_size))

# Предсказание для всей сетки
output = classifier.predict(np.c_[x_values.ravel(), y_values.ravel()])

# Визуализация границ
plt.figure()
plt.title('Границы классификации')
plt.pcolormesh(x_values, y_values, output.reshape(x_values.shape), cmap=cm.Paired, shading='auto')

# Наложение обучающих точек
for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], marker=mapper[i], s=75, edgecolors='black', facecolors='none')

plt.xlim(x_values.min(), x_values.max())
plt.ylim(y_values.min(), y_values.max())

# Тестовая точка данных
test_datapoint = [5.1, 3.6]
plt.figure()
plt.title('Тестовая точка данных')

# Точки обучающей выборки на графике
for i in range(X.shape[0]):
    plt.scatter(X[i, 0], X[i, 1], marker=mapper[i], s=75, edgecolors='black', facecolors='none')

# Отмечаем тестовую точку
plt.scatter(test_datapoint[0], test_datapoint[1], marker='x', linewidth=3, s=200, color='black')

# Извлечение индексов K ближайших соседей
indices = classifier.kneighbors([test_datapoint], n_neighbors=num_neighbors, return_distance=False)
indices = indices[0]

# Отображение K ближайших соседей
plt. figure ()
plt.title('K ближайших соседей')
for i in indices:
    plt.scatter(X[i, 0], X[i, 1], marker=marker_shapes[y[i] % len(marker_shapes)],
                linewidth=1, s=100, facecolors='black')

plt.scatter(test_datapoint[0], test_datapoint[1], marker='x', linewidth=6, s=200, facecolors='black')

for i in range(X.shape[0]):
     plt.scatter(X[i, 0], X[i, 1], marker=mapper[i], s=75, edgecolors='black', facecolors='none')

# Определение класса для тестовой точки
predicted_class = classifier.predict([test_datapoint])[0]

# Вывод результата
print(f'Предсказанный класс для точки {test_datapoint}: {predicted_class}')
plt.show()
