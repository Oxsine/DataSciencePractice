import sys

import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import ExtraTreesClassifier 
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report

from utilities import visualize_classifier

# Загрузка входных данных
input_file = 'data_imbalance.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Разделение входных данных на два класса по меткам
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])

# Визуализация входных данных
plt.figure()
plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='black', 
                edgecolors='black', linewidth=1, marker='x')
plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', 
                edgecolors='black', linewidth=1, marker='o')
plt.title('Входные данные')

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split( 
        X, y, test_size=0.25, random_state=5)

# Классификатор "Чрезвычайно случайных лесов"
params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
if len(sys.argv) > 1:
    if sys.argv[1] == 'balance':
        params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0, 'class_weight': 'balanced'}
    else:
        raise TypeError("Неверный аргумент командной строки; должен быть 'balance'")

classifier = ExtraTreesClassifier(**params)
classifier.fit(X_train, y_train)
visualize_classifier(classifier, X_train, y_train, 'Обучающая выборка')

y_test_pred = classifier.predict(X_test)
visualize_classifier(classifier, X_test, y_test, 'Тестовая выборка')

# Оценка производительности классификатора
class_names = ['Класс-0', 'Класс-1']
print("\n" + "#"*40)
print("\nПроизводительность классификатора на обучающей выборке\n")
print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
print("#"*40 + "\n")

print("#"*40)
print("\nПроизводительность классификатора на тестовой выборке\n")
print(classification_report(y_test, y_test_pred, target_names=class_names))
print("#"*40 + "\n")

plt.show()