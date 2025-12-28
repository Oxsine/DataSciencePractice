import numpy as np
import matplotlib.pyplot as plt
from matplotlib import patches
from sklearn import datasets
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import StratifiedKFold

# Загрузка набора данных iris
iris = datasets.load_iris()

# Разбиение данных на обучающий и тестовый наборы
# (в пропорции 80/20)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Используем первый набор
for train_index, test_index in skf.split(iris.data, iris.target):
    break  # Берем только первую итерацию

# Извлечем обучающие данные и метки
X_train = iris.data[train_index]
y_train = iris.target[train_index]

# Извлечем тестовые данные и метки
X_test = iris.data[test_index]
y_test = iris.target[test_index]

# Извлечение количества классов
num_classes = len(np.unique(y_train))

# Создание GMM
classifier = GaussianMixture(n_components=num_classes, 
                            covariance_type='full',
                            init_params='kmeans',
                            max_iter=20,
                            random_state=42)

# Инициализация средних GMM
classifier.means_init = np.array([X_train[y_train == i].mean(axis=0)
                                  for i in range(num_classes)])

# Обучение GMM-классификатора
classifier.fit(X_train)

# Создание графика
plt.figure(figsize=(12, 8))

# Выборка только первых двух признаков для визуализации
X_train_vis = X_train[:, :2]
X_test_vis = X_test[:, :2]
iris_data_vis = iris.data[:, :2]

# Обучение отдельной модели для визуализации (только на 2 признаках)
classifier_vis = GaussianMixture(n_components=num_classes, 
                                covariance_type='full',
                                init_params='kmeans',
                                max_iter=20,
                                random_state=42)

classifier_vis.means_init = np.array([X_train_vis[y_train == i].mean(axis=0)
                                      for i in range(num_classes)])
classifier_vis.fit(X_train_vis)

# Вычерчивание границ
colors = ['blue', 'green', 'red']

for i, color in enumerate(colors):
    # Извлечение собственных значений и собственных векторов
    eigenvalues, eigenvectors = np.linalg.eigh(
        classifier_vis.covariances_[i][:2, :2])
    
    # Нормализация первого собственного вектора
    norm_vec = eigenvectors[0] / np.linalg.norm(eigenvectors[0])
    
    # Извлечение угла наклона
    angle = np.arctan2(norm_vec[1], norm_vec[0])
    angle = 180 * angle / np.pi
    
    # Масштабный множитель для увеличения эллипсов
    scaling_factor = 8
    eigenvalues *= scaling_factor
    
    # Вычисление ширины и высоты эллипса (взятие квадратного корня)
    width = np.sqrt(eigenvalues[0])
    height = np.sqrt(eigenvalues[1])
    
    # Вычерчивание эллипсов
    ellipse = patches.Ellipse(
        xy=classifier_vis.means_[i, :2],
        width=2 * width,  # умножаем на 2 для полной ширины
        height=2 * height,  # умножаем на 2 для полной высоты
        angle=180 + angle, 
        color=color,
        alpha=0.3,
        linewidth=2
    )
    ellipse.set_clip_box(plt.gca().bbox)
    plt.gca().add_artist(ellipse)

# Отображение всех данных iris для контекста
for i, color in enumerate(colors):
    cur_data = iris_data_vis[iris.target == i]
    plt.scatter(cur_data[:, 0], cur_data[:, 1], 
                color=color, s=40, alpha=0.7,
                label=f'{iris.target_names[i]} (все данные)')

# Отображение тестовых данных
for i, color in enumerate(colors):
    test_data = X_test_vis[y_test == i]
    if len(test_data) > 0:  # Проверка, что есть данные для этого класса
        plt.scatter(test_data[:, 0], test_data[:, 1], 
                    color=color, marker='s', s=80,
                    edgecolors='black', linewidths=1.5,
                    label=f'{iris.target_names[i]} (тестовые)')

# Отображение обучающих данных
for i, color in enumerate(colors):
    train_data = X_train_vis[y_train == i]
    plt.scatter(train_data[:, 0], train_data[:, 1], 
                color=color, marker='o', s=60,
                edgecolors='black', linewidths=1,
                alpha=0.6,
                label=f'{iris.target_names[i]} (обучающие)')

# Вычисление прогнозных результатов для обучающих и тестовых данных
y_train_pred = classifier.predict(X_train)
accuracy_training = np.mean(y_train_pred == y_train) * 100
print(f'Точность на обучающих данных = {accuracy_training:.2f}%')

y_test_pred = classifier.predict(X_test)
accuracy_testing = np.mean(y_test_pred == y_test) * 100
print(f'Точность на тестовых данных = {accuracy_testing:.2f}%')

# Настройка графика
plt.title('GMM-классификатор на данных Iris')
plt.xlabel('Длина чашелистика (см)')
plt.ylabel('Ширина чашелистика (см)')
plt.legend(loc='best', fontsize=10)
plt.grid(True, alpha=0.3)

# Автоматическая настройка границ
x_min, x_max = iris_data_vis[:, 0].min() - 0.5, iris_data_vis[:, 0].max() + 0.5
y_min, y_max = iris_data_vis[:, 1].min() - 0.5, iris_data_vis[:, 1].max() + 0.5
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)

plt.tight_layout()
plt.show()

# Вывод дополнительной информации
print(f"\nРазмер обучающей выборки: {len(X_train)}")
print(f"Размер тестовой выборки: {len(X_test)}")
print(f"Количество классов: {num_classes}")

# Матрица ошибок
from sklearn.metrics import confusion_matrix, classification_report
print("\nМатрица ошибок для тестовых данных:")
cm = confusion_matrix(y_test, y_test_pred)
print(cm)

print("\nОтчет классификации:")
print(classification_report(y_test, y_test_pred, target_names=iris.target_names))

# Визуализация матрицы ошибок
fig, ax = plt.subplots(figsize=(6, 5))
im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
ax.figure.colorbar(im, ax=ax)
ax.set(xticks=np.arange(cm.shape[1]),
       yticks=np.arange(cm.shape[0]),
       xticklabels=iris.target_names,
       yticklabels=iris.target_names,
       title='Матрица ошибок',
       ylabel='Истинный класс',
       xlabel='Предсказанный класс')

# Добавление текста в ячейки
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")

plt.tight_layout()
plt.show()