from simpleai.search import CspProblem, backtrack

def constraint_func(names, values):
    return values[0] != values[1]  

if __name__ == '__main__':
    names = ('Марк', 'Джулия', 'Стив', 'Аманда', 'Брайан', 
             'Джоанн', 'Дерек', 'Аллан', 'Мишель', 'Келли')
    
    colors = dict((name, ['красный', 'зелёный', 'синий', 'серый']) for name in names)

    constraints = [
        (('Марк', 'Джулия'), constraint_func),
        (('Марк', 'Стив'), constraint_func),
        (('Джулия', 'Стив'), constraint_func),
        (('Джулия', 'Аманда'), constraint_func),
        (('Джулия', 'Дерек'), constraint_func),
        (('Джулия', 'Брайан'), constraint_func),
        (('Стив', 'Аманда'), constraint_func),
        (('Стив', 'Аллан'), constraint_func),
        (('Стив', 'Мишель'), constraint_func),
        (('Аманда', 'Мишель'), constraint_func),
        (('Аманда', 'Джоанн'), constraint_func),
        (('Аманда', 'Дерек'), constraint_func),
        (('Брайан', 'Дерек'), constraint_func),
        (('Брайан', 'Келли'), constraint_func),
        (('Джоанн', 'Мишель'), constraint_func),
        (('Джоанн', 'Аманда'), constraint_func),
        (('Джоанн', 'Дерек'), constraint_func),
        (('Джоанн', 'Келли'), constraint_func),
        (('Дерек', 'Келли'), constraint_func),
    ]

    problem = CspProblem(names, colors, constraints)

    output = backtrack(problem)
    print('\nРаспределение цветов:\n')
    for k, v in output.items():
        print(k, '==>', v)