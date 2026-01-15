import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier

# Загрузка входных данных
input_file = 'data_random_forests.txt'
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Разделение входных данных на три класса в соответствии с метками
class_0 = np.array(X[y==0])
class_1 = np.array(X[y==1])
class_2 = np.array(X[y==2])

# Разделение данных на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=5)

# Определение сетки параметров
parameter_grid = [ {'n_estimators': [100], 'max_depth': [2, 4, 7, 12, 16]},
                   {'max_depth': [4], 'n_estimators': [25, 50, 100, 250]}
                 ]

metrics = ['precision_weighted', 'recall_weighted']

for metric in metrics:
    print(f"\n##### Поиск оптимальных параметров для метрики: {metric}")

    classifier = GridSearchCV(
            ExtraTreesClassifier(random_state=0), 
            parameter_grid, cv=5, scoring=metric)
    classifier.fit(X_train, y_train)

    print("\nРезультаты кросс-валидации для сетки параметров:")
    for i, params in enumerate(classifier.cv_results_['params']):
        mean_score = classifier.cv_results_['mean_test_score'][i]
        print(params, '-->', round(mean_score, 3))

    print(f"\nЛучшие параметры: {classifier.best_params_}")

    y_pred = classifier.predict(X_test)
    print("\nОтчет о качестве классификации:\n")
    print(classification_report(y_test, y_pred))