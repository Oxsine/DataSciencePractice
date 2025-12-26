import argparse
import json
import numpy as np
from compute_scores import pearson_score
from collaborative_filtering import find_similar_users


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Рекомендуемые фильмы для указанного пользователя')
    parser.add_argument('--user', dest='user', required=True, help='Выбранный пользователь')
    return parser


# Получить рекомендации по фильмам для указанного пользователя
def get_recommendations(dataset, input_user):
    if input_user not in dataset:
        raise TypeError('Не удалось найти пользователя ' + input_user + ' в наборе данных')

    overall_scores = {}
    sirnilarity_scores = {}

    for user in (x for x in dataset if x != input_user):
        # Вычисляем коэффициент корреляции Пирсона между пользователями
        similarity_score = pearson_score(dataset, input_user, user)

        # Если схожесть равна или меньше 0, пропускаем
        if similarity_score <= 0:
            continue

        # Отфильтровываем фильмы, которых пользователь уже не оценивал
        filtered_list = [film for film in dataset[user] if
                         film not in dataset[input_user] or dataset[input_user][film] == 0]

        # Обновляем общие оценки и схожесть
        for film in filtered_list:
            overall_scores[film] = overall_scores.get(film, 0) + dataset[user][film] * similarity_score
            sirnilarity_scores[film] = sirnilarity_scores.get(film, 0) + similarity_score

    if not overall_scores:
        return ['Нет рекомендаций']

    # Генерируем рейтинги фильмов с помощью нормализации
    movie_scores = np.array([[score / sirnilarity_scoresи[film], film] for film, score in overall_scores.items()])

    # Сортируем по убыванию
    movie_scores_sorted = movie_scores[np.argsort(movie_scores[:, 0])][::-1]

    # Извлекаем фильмы в порядке рекомендаций
    movie_recommendations = [film for score, film in movie_scores_sorted]

    return movie_recommendations
if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    ratings_file = 'ratings.json'
    with open(ratings_file, 'r') as f:
        data = json.load(f)

    print("\nРекомендуемые фильмы для пользователя " + user + ":")
    movies = get_recommendations(data, user)
    for i, movie in enumerate(movies):
        print(str(i + 1) + '. ' + movie)
