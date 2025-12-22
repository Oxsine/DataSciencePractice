from simpleai.search import CspProblem, backtrack, \
        min_conflicts, MOST_CONSTRAINED_VARIABLE, \
        HIGHEST_DEGREE_VARIABLE, LEAST_CONSTRAINING_VALUE

def constraint_unique(variables, values):
    return len(values) == len(set(values))  

def constraint_bigger(variables, values):
    return values[0] > values[1]

def constraint_odd_even(variables, values):
    if values[0] % 2 == 0:
        return values[1] % 2 == 1 
    else:
        return values[1] % 2 == 0

if __name__ == '__main__':
    variables = ('Джон', 'Анна', 'Том', 'Патрисия')

    domains = {
        'Джон': [1, 2, 3, 4],
        'Анна': [1, 3, 5],
        'Том': [2, 4, 3],
        'Патрисия': [2, 3, 4, 5],
    }

    constraints = [
        (('Джон', 'Анна', 'Том'), constraint_unique), 
        (('Том', 'Анна'), constraint_bigger),          
        (('Джон', 'Патрисия'), constraint_odd_even),   
    ]

    problem = CspProblem(variables, domains, constraints)

    print('\nРешения:\n\nОбычный поиск:', backtrack(problem))
    print('\nНаиболее ограниченная переменная:', backtrack(problem, 
            variable_heuristic=MOST_CONSTRAINED_VARIABLE))
    print('\nПеременная с наибольшей степенью:', backtrack(problem, 
            variable_heuristic=HIGHEST_DEGREE_VARIABLE))
    print('\nНаименее ограничивающее значение:', backtrack(problem, 
            value_heuristic=LEAST_CONSTRAINING_VALUE))
    print('\nНаиболее ограниченная переменная и наименее ограничивающее значение:', 
            backtrack(problem, variable_heuristic=MOST_CONSTRAINED_VARIABLE, 
            value_heuristic=LEAST_CONSTRAINING_VALUE))
    print('\nНаибольшая степень и наименее ограничивающее значение:', 
            backtrack(problem, variable_heuristic=HIGHEST_DEGREE_VARIABLE, 
            value_heuristic=LEAST_CONSTRAINING_VALUE))
    print('\nМинимальные конфликты:', min_conflicts(problem))