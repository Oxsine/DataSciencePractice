import numpy as np

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
