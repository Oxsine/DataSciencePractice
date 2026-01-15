import random
from nltk import NaiveBayesClassifier
from nltk.classify import accuracy as nltk_accuracy
from nltk.corpus import names

# Извлечение последних N букв из входного слова
# и возврат значения, выступающего в качестве "признака"
def extract_features(word, N=2):
    last_n_letters = word[-N:]
    return {'feature': last_n_letters.lower()}

if __name__ == '__main__':
    # Создание обучающих данных с использованием
    # помеченных имен, доступных в NLTK
    male_list = [(name, 'male') for name in names.words('male.txt')]
    female_list = [(name, 'female') for name in names.words('female.txt')]
    data = (male_list + female_list)

    # Затравочное значение для генератора случайных чисел
    random.seed(5)

    # Перемешивание данных
    random.shuffle(data)

    # Создание тестовых данных
    input_names = ['Alexander', 'Danielle', 'David', 'Cheryl']

    # Определение количества образцов, используемых
    # для тренировки и тестирования
    num_train = int(0.8 * len(data))

    # Итерирование по различным длинам конечного
    # фрагмента для сравнения точности
    for i in range(1, 6):
        print('\nNumber of end letters:', i)
        features = [(extract_features(n, i), gender) for (n, gender) in data]

        train_data, test_data = features[:num_train], features[num_train:]

        classifier = NaiveBayesClassifier.train(train_data)

        # Вычисление точности классификатора
        accuracy = round(100 * nltk_accuracy(classifier, test_data), 2)
        print('Accuracy = ' + str(accuracy) + '%')

        # Предсказание результатов для входных имён
        # с использованием обученной модели классификатора
        for name in input_names:
            result = classifier.classify(extract_features(name, i))
            print(name, '==>', result)
