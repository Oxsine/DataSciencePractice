import argparse
import json
import numpy as np
from compute_scores import pearson_score

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Найти пользователей, которые похожи на указанного пользователя')
    parser.add_argument('--user', dest='user', required=True, help='Указанный пользователь')
    return parser

# Поиск в наборе данных пользователей, аналогичных указанному
def find_similar_users(dataset, user, num_users):
    if user not in dataset:
        raise TypeError('Не удалось найти пользователя ' + user + ' в наборе данных')

    # Вычисление оценки сходства по Пирсону между указанным пользователем
    scores = np.array([[x, pearson_score(dataset, user, x)] for x in dataset if x != user])

    # Сортировка оценок по убыванию
    scores_sorted = scores[np.argsort(scores[:, 1])][::-1]

    # Извлечение первых 'num_users' пользователей
    top_users = scores_sorted[:num_users]
    return top_users


if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    ratings_file = 'ratings.json'
    with open(ratings_file, 'r') as f:
        data = json.load(f)

    print('\nПользователи, похожие на ' + user + ':\n')
    similar_users = find_similar_users(data, user, 3)
    print('\nПохожие пользователи:')
    print('Пользователь\t\tОценка сходства')
    print('-' * 40)
    for item in similar_users:
        print(item[0], '\t\t', round(float(item[1]), 2))
