import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y):
    # Определение диапазонов для осей X и Y
    x_min, x_max = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    y_min, y_max = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Размер ячейки сетки
    step = 0.01

    # Формирование координатной решетки
    grid_x, grid_y = np.meshgrid(
        np.arange(x_min, x_max, step),
        np.arange(y_min, y_max, step)
    )

    # Прогнозирование на всех точках сетки
    result = classifier.predict(np.c_[grid_x.ravel(), grid_y.ravel()])

    # Приведение к форме матрицы
    result = result.reshape(grid_x.shape)

    # Создание графического окна
    plt.figure()

    # Отрисовка областей решения
    plt.pcolormesh(grid_x, grid_y, result, cmap=plt.cm.gray)

    # Отображение исходных точек
    plt.scatter(X[:, 0], X[:, 1], c=y, s=80, edgecolors='gray', linewidth=1.5, cmap=plt.cm.Paired)

    # Установка пределов осей
    plt.xlim(grid_x.min(), grid_x.max())
    plt.ylim(grid_y.min(), grid_y.max())

    # Настройка делений шкал
    plt.xticks((np.arange(int(X[:, 0].min() - 1), int(X[:, 0].max() + 1), 1.0)))
    plt.yticks((np.arange(int(X[:, 1].min() - 1), int(X[:, 1].max() + 1), 1.0)))

    plt.show()