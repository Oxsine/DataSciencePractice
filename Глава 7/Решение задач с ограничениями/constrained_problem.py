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

if __name__=='__main__':
    variables = ('Иван', 'Аня', 'Толик', 'Павел')

    domains = {
        'Иван': [1, 2, 3],
        'Аня': [1, 3],
        'Толик': [2, 4],
        'Павел': [2, 3, 4],
    }

    constraints = [
        (('Иван', 'Аня', 'Толик'), constraint_unique),
        (('Толик', 'Аня'), constraint_bigger),
        (('Иван', 'Павел'), constraint_odd_even),
    ]

    problem = CspProblem(variables, domains, constraints)

    print('\nРешение:\n\nНормальное:', backtrack(problem))
    print('\nНаиболее ограниченная переменная:', backtrack(problem, 
            variable_heuristic=MOST_CONSTRAINED_VARIABLE))
    print('\nПеременная наивысшей степени:', backtrack(problem, 
            variable_heuristic=HIGHEST_DEGREE_VARIABLE))
    print('\nНаименьшее ограничивающее значение:', backtrack(problem, 
            value_heuristic=LEAST_CONSTRAINING_VALUE))
    print('\nНаиболее ограничивающая переменная и наименее ограничивающее значение:', 
            backtrack(problem, variable_heuristic=MOST_CONSTRAINED_VARIABLE, 
            value_heuristic=LEAST_CONSTRAINING_VALUE))
    print('\nНаивысшая степень и наименьшее ограничивающее значение:', 
            backtrack(problem, variable_heuristic=HIGHEST_DEGREE_VARIABLE, 
            value_heuristic=LEAST_CONSTRAINING_VALUE))
    print('\nМинимум конфликтов:', min_conflicts(problem))
