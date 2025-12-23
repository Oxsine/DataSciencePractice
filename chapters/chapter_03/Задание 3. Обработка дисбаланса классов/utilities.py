import numpy as np
import matplotlib.pyplot as plt

def visualize_classifier(classifier, X, y, title=''):
    # Определение минимальных и максимальных значений X и Y
    # которые будут использоваться для создания сетки
    min_x, max_x = X[:, 0].min() - 1.0, X[:, 0].max() + 1.0
    min_y, max_y = X[:, 1].min() - 1.0, X[:, 1].max() + 1.0

    # Определение шага для построения сетки
    mesh_step_size = 0.01

    # Создание сетки значений X и Y
    x_vals, y_vals = np.meshgrid(
        np.arange(min_x, max_x, mesh_step_size),
        np.arange(min_y, max_y, mesh_step_size)
    )

    # Применение классификатора к точкам сетки
    output = classifier.predict(np.c_[x_vals.ravel(), y_vals.ravel()])

    # Изменение формы выходного массива
    output = output.reshape(x_vals.shape)

    # Создание графика
    plt.figure()

    # Установка заголовка
    plt.title(title)

    # Выбор цветовой схемы для графика
    plt.pcolormesh(x_vals, y_vals, output, cmap=plt.cm.gray)

    # Наложение обучающих точек на график
    plt.scatter(
        X[:, 0], X[:, 1], 
        c=y, s=75, edgecolors='black', 
        linewidth=1, cmap=plt.cm.Paired
    )

    # Установка границ графика
    plt.xlim(x_vals.min(), x_vals.max())
    plt.ylim(y_vals.min(), y_vals.max())

    # Установка делений на осях X и Y
    plt.xticks((
        np.arange(
            int(X[:, 0].min() - 1), 
            int(X[:, 0].max() + 1), 
            1.0
        )
    ))
    plt.yticks((
        np.arange(
            int(X[:, 1].min() - 1), 
            int(X[:, 1].max() + 1), 
            1.0
        )
    ))

    plt.show()