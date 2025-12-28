import argparse
import json
import numpy as np

def build_arg_parser():
    parser = argparse.ArgumentParser(description='Вычисление оценки сходства')
    parser.add_argument('--user1', dest='user1', required=True, help='Первый пользователь')
    parser.add_argument('--user2', dest='user2', required=True, help='Второй пользователь')
    parser.add_argument("--score-type", dest="score_type", required=True, 
                       choices=['Euclidean', 'Pearson'], help='Метрика сходства для использования')
    return parser

def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Не найден пользователь ' + user1 + ' в данных')
    if user2 not in dataset:
        raise TypeError('Не найден пользователь ' + user2 + ' в данных')

    common_movies = []
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies.append(item)

    if len(common_movies) == 0:
        return 0

    squared_diff = []
    for item in common_movies:
        squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))
    return 1 / (1 + np.sqrt(np.sum(squared_diff)))

def pearson_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Не найден пользователь ' + user1 + ' в данных')
    if user2 not in dataset:
        raise TypeError('Не найден пользователь ' + user2 + ' в данных')

    common_movies = []
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies.append(item)

    n = len(common_movies)
    if n == 0:
        return 0

    sum1 = sum([dataset[user1][it] for it in common_movies])
    sum2 = sum([dataset[user2][it] for it in common_movies])

    sum1_sq = sum([np.square(dataset[user1][it]) for it in common_movies])
    sum2_sq = sum([np.square(dataset[user2][it]) for it in common_movies])

    product_sum = sum([dataset[user1][it] * dataset[user2][it] for it in common_movies])

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

    # Загрузка данных рейтингов из файла с указанием кодировки
    ratings_file = '../../data/ratings.json'
    
    # Пробуем разные кодировки
    encodings_to_try = ['utf-8', 'utf-8-sig', 'cp1251', 'iso-8859-1', 'windows-1252']
    data = None
    
    for encoding in encodings_to_try:
        try:
            with open(ratings_file, 'r', encoding=encoding) as f:
                data = json.load(f)
            print(f"Файл успешно загружен с кодировкой: {encoding}")
            break
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            print(f"Не удалось загрузить с кодировкой {encoding}: {e}")
            continue
    
    if data is None:
        # Если ни одна кодировка не сработала, пробуем бинарный режим
        try:
            with open(ratings_file, 'rb') as f:
                content = f.read()
                # Пробуем декодировать с разными кодировками
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

    # Выбор метрики
    try:
        if score_type == 'Euclidean':
            score = euclidean_score(data, user1, user2)
            print('Показатель Евклидова расстояния:')
            print(f'{score:.4f}')
        else:
            score = pearson_score(data, user1, user2)
            print('Коэффициент корреляции Пирсона:')
            print(f'{score:.4f}')
    except TypeError as e:
        print(f"Ошибка: {e}")
        print("\nДоступные пользователи:")
        for user in data.keys():
            print(f"  {user}")
        exit(1)
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")
        exit(1)