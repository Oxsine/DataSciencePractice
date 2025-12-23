from simpleai.search import astar, SearchProblem  # Импорт алгоритма A* и базового класса для задач поиска
    
class PuzzleSolver(SearchProblem):
    """Класс для решения головоломки 'Пятнашки' (3x3) с использованием алгоритма A*"""
    
    def actions(self, cur_state):
        """
        Возвращает возможные действия (ходы) из текущего состояния.
        Действие - это число, которое можно передвинуть на пустую клетку ('e').
        """
        rows = string_to_list(cur_state)  # Преобразуем строку состояния в список строк
        row_empty, col_empty = get_location(rows, 'e')  # Находим координаты пустой клетки

        actions = []
        # Проверяем возможность перемещения чисел в пустую клетку со всех четырех сторон
        if row_empty > 0:  # Можно переместить число сверху
            actions.append(rows[row_empty - 1][col_empty])
        if row_empty < 2:  # Можно переместить число снизу
            actions.append(rows[row_empty + 1][col_empty])
        if col_empty > 0:  # Можно переместить число слева
            actions.append(rows[row_empty][col_empty - 1])
        if col_empty < 2:  # Можно переместить число справа
            actions.append(rows[row_empty][col_empty + 1])

        return actions

    def result(self, state, action):
        """
        Применяет действие (перемещает число) к текущему состоянию
        и возвращает новое состояние.
        """
        rows = string_to_list(state)  # Преобразуем строку в список
        row_empty, col_empty = get_location(rows, 'e')  # Находим пустую клетку
        row_new, col_new = get_location(rows, action)  # Находим клетку с числом для перемещения

        # Меняем местами пустую клетку и число
        rows[row_empty][col_empty], rows[row_new][col_new] = \
                rows[row_new][col_new], rows[row_empty][col_empty]

        return list_to_string(rows)  # Возвращаем новое состояние в строковом формате

    def is_goal(self, state):
        """Проверяет, является ли текущее состояние целевым"""
        return state == GOAL

    def heuristic(self, state):
        """
        Эвристическая функция для алгоритма A*.
        Вычисляет манхэттенское расстояние - сумму расстояний каждой плитки
        от ее текущей позиции до целевой позиции.
        """
        rows = string_to_list(state)  # Преобразуем строку в список

        distance = 0  # Суммарное расстояние

        # Для каждой плитки (1-8) и пустой клетки ('e')
        for number in '12345678e':
            row_new, col_new = get_location(rows, number)  # Текущая позиция плитки
            row_new_goal, col_new_goal = goal_positions[number]  # Целевая позиция плитки

            # Суммируем манхэттенское расстояние (разница по строкам + разница по столбцам)
            distance += abs(row_new - row_new_goal) + abs(col_new - col_new_goal)

        return distance

# Вспомогательные функции для преобразования между форматами

def list_to_string(input_list):
    """Преобразует список списков (матрицу 3x3) в строку с разделителями"""
    return '\n'.join(['-'.join(x) for x in input_list])

def string_to_list(input_string):
    """Преобразует строку с разделителями в список списков (матрицу 3x3)"""
    return [x.split('-') for x in input_string.split('\n')]
 
def get_location(rows, input_element):
    """Находит координаты элемента в матрице"""
    for i, row in enumerate(rows):
        for j, item in enumerate(row):
            if item == input_element:
                return i, j  # Возвращаем (строка, столбец)

# Определение целевого и начального состояний головоломки

GOAL = '''1-2-3
4-5-6
7-8-e'''  # Целевое состояние (решённая головоломка)

INITIAL = '''1-e-2
6-3-4
7-5-8'''  # Начальное состояние (перемешанная головоломка)

# Создание словаря целевых позиций для каждой плитки
# Необходимо для быстрого вычисления эвристики

goal_positions = {}  # Словарь: плитка -> (целевая_строка, целевой_столбец)
rows_goal = string_to_list(GOAL)  # Преобразуем цель в список

# Заполняем словарь целевыми позициями для каждой плитки (1-8) и пустой клетки ('e')
for number in '12345678e':
    goal_positions[number] = get_location(rows_goal, number)

# Запускаем алгоритм A* для решения головоломки
result = astar(PuzzleSolver(INITIAL))

# Выводим путь решения
for i, (action, state) in enumerate(result.path()):
    print()  # Пустая строка для разделения шагов
    
    # Выводим описание действия в зависимости от позиции в пути
    if action == None:
        print('Начальная конфигурация')
    elif i == len(result.path()) - 1:
        print('После передвижения', action, 'в пустую клетку. Цель достигнута!')
    else:
        print('После передвижения', action, 'в пустую клетку')

    print(state)  # Выводим текущее состояние головоломки