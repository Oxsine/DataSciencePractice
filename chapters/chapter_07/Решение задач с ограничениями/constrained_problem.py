from simpleai.search import CspProblem, backtrack, \
        min_conflicts, MOST_CONSTRAINED_VARIABLE, \
        HIGHEST_DEGREE_VARIABLE, LEAST_CONSTRAINING_VALUE

# Определение функций-ограничений

def constraint_unique(variables, values):
    """Ограничение уникальности: все значения должны быть разными"""
    return len(values) == len(set(values))  # Проверяем, что нет дубликатов

def constraint_bigger(variables, values):
    """Ограничение сравнения: первое значение должно быть больше второго"""
    return values[0] > values[1]  # Захар > Анны

def constraint_odd_even(variables, values):
    """Ограничение четности: если первое число четное, второе должно быть нечетным, и наоборот"""
    if values[0] % 2 == 0:
        return values[1] % 2 == 1  # Михаил четный → Тамар нечетная
    else:
        return values[1] % 2 == 0  # Михаил нечетный → Тамар четная

if __name__ == '__main__':
    # Определение переменных задачи
    variables = ('Михаил', 'Катя', 'Захар', 'Тамар')

    # Домены значений для каждой переменной (возможные значения)
    domains = {
        'Михаил': [1, 2, 3, 4],
        'Катя': [1, 3, 5],        # Только нечетные, кроме 5
        'Захар': [2, 4, 3],         # В основном четные + 3
        'Тамар': [2, 3, 4, 5], # Все доступные значения
    }

    # Определение ограничений между переменными
    constraints = [
        # Тройное ограничение: Михаил, Катя и Захар должны иметь разные значения
        (('Михаил', 'Катя', 'Захар'), constraint_unique), 
        # Парное ограничение: значение Захара должно быть больше значения Анны
        (('Захар', 'Катя'), constraint_bigger),          
        # Парное ограничение: Михаил и Тамар должны иметь разную четность
        (('Михаил', 'Тамар'), constraint_odd_even),   
    ]

    # Создание объекта задачи удовлетворения ограничений (CSP)
    problem = CspProblem(variables, domains, constraints)

    # Решение задачи различными методами и вывод результатов

    # 1. Стандартный backtracking (обратный отсчет) без эвристик
    print('\nРешения:\n\nОбычный поиск:', backtrack(problem))
    
    # 2. Backtracking с эвристикой "наиболее ограниченная переменная"
    # (выбирается переменная с наименьшим количеством доступных значений)
    print('\nНаиболее ограниченная переменная:', backtrack(problem, 
            variable_heuristic=MOST_CONSTRAINED_VARIABLE))
    
    # 3. Backtracking с эвристикой "переменная с наибольшей степенью"
    # (выбирается переменная, участвующая в наибольшем количестве ограничений)
    print('\nПеременная с наибольшей степенью:', backtrack(problem, 
            variable_heuristic=HIGHEST_DEGREE_VARIABLE))
    
    # 4. Backtracking с эвристикой "наименее ограничивающее значение"
    # (выбирается значение, которое оставляет максимальную свободу другим переменным)
    print('\nНаименее ограничивающее значение:', backtrack(problem, 
            value_heuristic=LEAST_CONSTRAINING_VALUE))
    
    # 5. Комбинация: наиболее ограниченная переменная + наименее ограничивающее значение
    print('\nНаиболее ограниченная переменная и наименее ограничивающее значение:', 
            backtrack(problem, variable_heuristic=MOST_CONSTRAINED_VARIABLE, 
            value_heuristic=LEAST_CONSTRAINING_VALUE))
    
    # 6. Комбинация: переменная с наибольшей степенью + наименее ограничивающее значение
    print('\nНаибольшая степень и наименее ограничивающее значение:', 
            backtrack(problem, variable_heuristic=HIGHEST_DEGREE_VARIABLE, 
            value_heuristic=LEAST_CONSTRAINING_VALUE))
    
    # 7. Алгоритм минимальных конфликтов (min-conflicts)
    # (локальный поиск, который пытается минимизировать количество конфликтов)
    print('\nМинимальные конфликты:', min_conflicts(problem))