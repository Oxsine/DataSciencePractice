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
indices = StratifiedKFold(iris.target, n_splits=5)

# Используем первый набор
train_index, test_index = next(iter(indices))

# Извлечем обучающие данные и метки
Х_train_iris.data[train_index]
y_train = iris.target[train_index]

# Извлечем тестовые данные и метки
Х_test_iris.data[test_index]
y_test = iris.target[test_index]

# Извлечение количества классов
num_classes = len(np.unique(y_train))

# Создание GММ
classifier_GММ(n_components=num_classes, covariance_type='full',
   init_params='wc', n_iter=20)

# Инициализация средних GММ
classifier.means_np.array( [X_train[y_train == i] .mean(axis=O)
   for i in range(num_classes)])

# Обучение GММ-классификатора
classifier.fit(X_train)

# Вычерчивание границ
plt. figure ()
colors = 'bgr'
for i, color in enumerate(colors):
   # Извлечение собственных значений и собственных векторов
   eigenvalues, eigenvectors = np.linalg.eigh (
   classifier._get_covars() [i] [:2, :2])

# Нормализация первого собственного вектора
norm_vec = eigenvectors[O] / np.linalg.norm(eigenvectors[O])

# Извлечение угла наклона
angle_np.arctan2(norm_vec[l], norm_vec[O])
angle = 180 * angle / np.pi

# Масштабный множитель дпя увеличения эллипсов
# (выбрано произвольное значение, которое нас удовлетворяет)
scaling_factor = 8
eigenvalues *= scaling_factor

# Вычерчивание эллипсов
ellipse = patches.Ellipse(classifier.means_[i, :2],
eigenvalues[O], eigenvalues[l], 180 + angle,  color=color)
axis_handle = plt.subplot(l, 1, 1)
ellipse.set_clip_box(axis_hand.bbox)
ellipse.set_alpha(0.6)
axis_handle.add_artist(ellipse)

# Откладывание входных данных на графике
colors = 'bgr'
for i, color in enumerate(colors):
   cur_data = iris.data[iris.target == i]
   plt.scatter(cur_data[:,0], cur_data[:,1], marker='o',
      facecolors='none', edgecolors='Ьlack', s=40,
      label=iris.target_names[i])

test_data = X_test[y_test == i]
plt.scatter(test_data[:,O], test_data[:,1], marker='s',
facecolors='black', edgecolors='black', s=40,
label=iris.target_names[i])

# Вычисление прогнозных результатов для обучающих и тестовых данных
y_train_pred = classifier.predict(X_train)
accuracy_training = np.mean(y_train_pred.ravel(),
y_train.ravel()) * 100
print('Точность на тестовых данных =', accuracy_training)
y_test_pred = classifier.predict(X_test)
accuracy_testing = np.mean(y_test_pred.ravel(),
y_test.ravel()) * 100
print('Точность на тестовых данных =', accuracy_testing)
plt.title ('GММ-классификатор')
plt.xticks (( ))
plt.yticks (( ))
plt.show ( )

