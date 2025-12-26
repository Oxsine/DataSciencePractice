import argparse
import json
import numpy as np

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Вычисление оценки сходства')
    parser.add_argument('--user1', dest='user1', required=True, help='Первый пользователь')
    parser.add_argument('--user2', dest='user2', required=True, help='Второй пользователь')
    parser.add_argument("--score-type", dest="score_type", required=True, choices=['Euclidean', 'Pearson'], help='Метрика сходства для использования')
    return parser

# Введение функции вычисления евклидова расстояния между пользователями user1 и user2
def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Не найден пользователь ' + user1 + ' в данных')
    if user2 not in dataset:
        raise TypeError('Не найден пользователь ' + user2 + ' в данных')

    # Фильмы, оценённые обоими пользователями
    common_movies = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies.append(item)

    # В отсутствие фильмов, оцененных обоими пользователями,
    # оценка принимается равной О
    if len(common_movies) == 0:
        return 0

    # Расчёт евклидова расстояния
    squared_diff = []
    for item in common_movies:
        squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))
    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


# Введение функции вычисления коэффициента корреляции Пирсона
def pearson_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Не найден пользователь ' + user1 + ' в данных')
    if user2 not in dataset:
        raise TypeError('Не найден пользователь ' + user2 + ' в данных')

    # Общие фильмы, оценённые обоими пользователями
    common_movies = []

    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies.append(item)

    n = len(common_movies)

    # Если нет общих фильмов, результат 0
    if n == 0:
        return 0

    # Суммы оценок
    sum1 = sum([dataset[user1][it] for it in common_movies])
    sum2 = sum([dataset[user2][it] for it in common_movies])

    # Сумма квадратов оценок
    sum1_sq = sum([np.square(dataset[user1][it]) for it in common_movies])
    sum2_sq = sum([np.square(dataset[user2][it]) for it in common_movies])

    # Сумма произведений оценок
    product_sum = sum([dataset[user1][it] * dataset[user2][it] for it in common_movies])

    # Вычисляемай коэффициент Пирсона
    numerator = product_sum - (sum1 * sum2 / n)
    denominator = np.sqrt(
        (sum1_sq - np.square(sum1) / n) * (sum2_sq - np.square(sum2) / n)
    )

    if denominator == 0:
        return 0

    return numerator / denominator

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user1 = args.user1
    user2 = args.user2
    score_type = args.score_type

    # Загрузка данных рейтингов из файла
    ratings_file = 'ratings.json'
    with open(ratings_file, 'r') as f:
        data = json.load(f)

    # Выбор метрики
    if score_type == 'Euclidean':
        print('Показатель Евклидова расстояния:')
        print(euclidean_score(data, user1, user2))
    else:
        print('Коэффициент корреляции Пирсона:')
        print(pearson_score(data, user1, user2))
