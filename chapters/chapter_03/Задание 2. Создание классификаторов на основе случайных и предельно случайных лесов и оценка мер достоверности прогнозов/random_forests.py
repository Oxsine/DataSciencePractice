import argparse 

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.model_selection import train_test_split

from utilities import visualize_classifier

# Парсер аргументов командной строки
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Классификация данных с использованием методов ансамблевого обучения')
    parser.add_argument('--classifier-type', dest='classifier_type', 
            required=True, choices=['rf', 'erf'], help="Тип классификатора для использования; может быть 'rf' (Random Forest) или 'erf' (Extra Trees)")
    return parser

if __name__=='__main__':
    # Парсинг входных аргументов
    args = build_arg_parser().parse_args()
    classifier_type = args.classifier_type

    # Загрузка входных данных
    input_file = 'data_random_forests.txt'
    data = np.loadtxt(input_file, delimiter=',')
    X, y = data[:, :-1], data[:, -1]

    # Разделение входных данных на три класса по меткам
    class_0 = np.array(X[y==0])
    class_1 = np.array(X[y==1])
    class_2 = np.array(X[y==2])

    # Визуализация входных данных
    plt.figure()
    plt.scatter(class_0[:, 0], class_0[:, 1], s=75, facecolors='white', 
                    edgecolors='black', linewidth=1, marker='s')
    plt.scatter(class_1[:, 0], class_1[:, 1], s=75, facecolors='white', 
                    edgecolors='black', linewidth=1, marker='o')
    plt.scatter(class_2[:, 0], class_2[:, 1], s=75, facecolors='white', 
                    edgecolors='black', linewidth=1, marker='^')
    plt.title('Входные данные')

    # Разделение данных на обучающую и тестовую выборки
    X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=5)

    # Классификатор ансамблевого обучения
    params = {'n_estimators': 100, 'max_depth': 4, 'random_state': 0}
    if classifier_type == 'rf':
        classifier = RandomForestClassifier(**params)
    else:
        classifier = ExtraTreesClassifier(**params)

    classifier.fit(X_train, y_train)
    visualize_classifier(classifier, X_train, y_train, 'Обучающая выборка')

    y_test_pred = classifier.predict(X_test)
    visualize_classifier(classifier, X_test, y_test, 'Тестовая выборка')

    # Оценка производительности классификатора
    class_names = ['Класс-0', 'Класс-1', 'Класс-2']
    print("\n" + "#"*40)
    print("\nПроизводительность классификатора на обучающей выборке\n")
    print(classification_report(y_train, classifier.predict(X_train), target_names=class_names))
    print("#"*40 + "\n")

    print("#"*40)
    print("\nПроизводительность классификатора на тестовой выборке\n")
    print(classification_report(y_test, y_test_pred, target_names=class_names))
    print("#"*40 + "\n")

    # Вычисление уверенности классификации
    test_datapoints = np.array([[6, 5], [1, 6], [2, 4], [7, 4], [3, 4], [1, 3]])

    print("\nМера уверности классификации:")
    for datapoint in test_datapoints:
        probabilities = classifier.predict_proba([datapoint])[0]
        predicted_class = 'Класс-' + str(np.argmax(probabilities))
        print('\nТочка данных:', datapoint)
        print('Предсказанный класс:', predicted_class)

    # Визуализация тестовых точек данных
    visualize_classifier(classifier, test_datapoints, [0]*len(test_datapoints), 
            'Тестовые точки данных')

    plt.show()