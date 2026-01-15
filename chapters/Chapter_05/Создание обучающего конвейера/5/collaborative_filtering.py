import argparse
import json
import numpy as np
from utils import pearson_score


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

    ratings_file = '../../data/ratings.json'
    
    # Пробуем разные кодировки для чтения файла
    encodings_to_try = ['utf-8-sig', 'utf-8', 'cp1251', 'iso-8859-1']
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
        print('\nПользователи, похожие на ' + user + ':\n')
        similar_users = find_similar_users(data, user, 3)
        print('Похожие пользователи:')
        print('Пользователь\t\tОценка сходства')
        print('-' * 40)
        for item in similar_users:
            username = item[0]
            score = float(item[1])
            # Форматируем вывод для лучшей читаемости
            print(f'{username[:20]:20}\t{score:7.4f}')
    except TypeError as e:
        print(f"Ошибка: {e}")
        print("\nДоступные пользователи в базе данных:")
        print("-" * 40)
        for i, username in enumerate(data.keys(), 1):
            print(f"{i:2}. {username}")
    except Exception as e:
        print(f"Неизвестная ошибка: {e}")