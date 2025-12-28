import argparse
import json
import numpy as np
from utils import *

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
    # оценка принимается равной 0
    if len(common_movies) == 0:
        return 0

    # Расчёт евклидова расстояния
    squared_diff = []
    for item in common_movies:
        squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))
    return 1 / (1 + np.sqrt(np.sum(squared_diff)))


def build_arg_parser():
    parser = argparse.ArgumentParser(description='Рекомендуемые фильмы для указанного пользователя')
    parser.add_argument('--user', dest='user', required=True, help='Выбранный пользователь')
    return parser


# Получить рекомендации по фильмам для указанного пользователя
def get_recommendations(dataset, input_user):
    if input_user not in dataset:
        raise TypeError('Не удалось найти пользователя ' + input_user + ' в наборе данных')

    overall_scores = {}
    similarity_scores = {}

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
            similarity_scores[film] = similarity_scores.get(film, 0) + similarity_score

    if not overall_scores:
        return ['Нет рекомендаций']

    # Генерируем рейтинги фильмов с помощью нормализации
    movie_scores = np.array([[score / similarity_scores[film], film] for film, score in overall_scores.items()])

    # Сортируем по убыванию
    movie_scores_sorted = movie_scores[np.argsort(movie_scores[:, 0])][::-1]

    # Извлекаем фильмы в порядке рекомендаций
    movie_recommendations = [film for score, film in movie_scores_sorted]

    return movie_recommendations

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user = args.user

    ratings_file = '../../data/ratings.json'
    
    # Пробуем разные кодировки для чтения файла
    encodings_to_try = ['utf-8-sig', 'utf-8', 'cp1251', 'iso-8859-1', 'windows-1252']
    data = None
    
    for encoding in encodings_to_try:
        try:
            with open(ratings_file, 'r', encoding=encoding) as f:
                data = json.load(f)
            print(f"Файл успешно загружен с кодировкой: {encoding}")
            break
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"Не удалось загрузить с кодировкой {encoding}: {str(e)[:50]}...")
            continue
    
    # Если ни одна кодировка не сработала, пробуем бинарный режим
    if data is None:
        try:
            with open(ratings_file, 'rb') as f:
                content = f.read()
                for encoding in encodings_to_try:
                    try:
                        text = content.decode(encoding)
                        data = json.loads(text)
                        print(f"Файл загружен с кодировкой: {encoding}")
                        break
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        continue
        except Exception as e:
            print(f"Ошибка при загрузке файла: {e}")
            exit(1)
    
    if data is None:
        print("Не удалось загрузить файл с любой из попробованных кодировок.")
        exit(1)

    try:
        print(f"\nРекомендуемые фильмы для пользователя {user}:")
        movies = get_recommendations(data, user)
        if movies[0] == 'Нет рекомендаций':
            print("  " + movies[0])
        else:
            for i, movie in enumerate(movies, 1):
                print(f"  {i}. {movie}")
    except TypeError as e:
        print(f"Ошибка: {e}")
        print("\nДоступные пользователи в базе данных:")
        print("-" * 40)
        for i, username in enumerate(data.keys(), 1):
            num_movies = len(data[username])
            print(f"{i:2}. {username} ({num_movies} фильмов)")
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")