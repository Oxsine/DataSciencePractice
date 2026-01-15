import math
from simpleai.search import SearchProblem, astar

class MazeSolver(SearchProblem):
    """Класс для решения задачи поиска пути в лабиринте с помощью алгоритма A*"""
    
    def __init__(self, board):
        """
        Инициализация решателя лабиринта.
        
        Args:
            board: Двумерный список (матрица) символов, представляющий лабиринт
        """
        self.board = board  # Карта лабиринта
        self.goal = (0, 0)  # Целевая позиция (по умолчанию)

        # Поиск стартовой ('o') и целевой ('x') позиций на карте
        for y in range(len(self.board)):
            for x in range(len(self.board[y])):
                if self.board[y][x].lower() == "o":
                    self.initial = (x, y)  # Начальная позиция (координаты x, y)
                elif self.board[y][x].lower() == "x":
                    self.goal = (x, y)  # Целевая позиция

        # Инициализация родительского класса с начальным состоянием
        super(MazeSolver, self).__init__(initial_state=self.initial)

    def actions(self, state):
        """
        Возвращает список возможных действий из текущего состояния.
        Действие - это перемещение в соседнюю клетку.
        
        Args:
            state: Текущая позиция (x, y)
            
        Returns:
            Список разрешенных направлений движения
        """
        actions = []
        # Проверяем все возможные направления движения из словаря COSTS
        for action in COSTS.keys():
            newx, newy = self.result(state, action)  # Вычисляем новую позицию
            # Если новая позиция не является стеной ('#'), добавляем действие
            if self.board[newy][newx] != "#":
                actions.append(action)

        return actions

    def result(self, state, action):
        """
        Применяет действие (перемещение) к текущему состоянию.
        
        Args:
            state: Текущая позиция (x, y)
            action: Направление движения (например, "up", "down left")
            
        Returns:
            Новая позиция после выполнения действия
        """
        x, y = state  # Распаковываем текущие координаты

        # Обновляем координаты в зависимости от направления движения
        if action.count("up"):
            y -= 1  # Движение вверх уменьшает y
        if action.count("down"):
            y += 1  # Движение вниз увеличивает y
        if action.count("left"):
            x -= 1  # Движение влево уменьшает x
        if action.count("right"):
            x += 1  # Движение вправо увеличивает x

        new_state = (x, y)  # Формируем новое состояние

        return new_state

    def is_goal(self, state):
        """
        Проверяет, достигнута ли целевая позиция.
        
        Args:
            state: Текущая позиция (x, y)
            
        Returns:
            True, если текущая позиция совпадает с целевой, иначе False
        """
        return state == self.goal

    def cost(self, state, action, state2):
        """
        Возвращает стоимость выполнения действия.
        В данном случае используется фиксированная стоимость из словаря COSTS.
        
        Args:
            state: Начальное состояние
            action: Выполняемое действие
            state2: Конечное состояние
            
        Returns:
            Стоимость перемещения
        """
        return COSTS[action]

    def heuristic(self, state):
        """
        Эвристическая функция для алгоритма A*.
        Использует евклидово расстояние до цели.
        
        Args:
            state: Текущая позиция (x, y)
            
        Returns:
            Евклидово расстояние до цели
        """
        x, y = state  # Текущие координаты
        gx, gy = self.goal  # Целевые координаты

        # Вычисляем евклидово расстояние по формуле sqrt((x2-x1)² + (y2-y1)²)
        return math.sqrt((x - gx) ** 2 + (y - gy) ** 2)

if __name__ == "__main__":
    # Определение карты лабиринта в виде строки
    MAP = """
    ##############################
    #         #                  #
    # ####    ########       #   #
    #  o #    #              #   #
    #    ###     #############   #
    #      #   ###   #           #
    #      #     #   ####  #   ###
    #     #####    #    #### x   #
    #              #       #     #
    ##############################
    """

    # Вывод исходной карты лабиринта
    print(MAP)
    # Преобразование строки карты в двумерный список символов
    # split("\n") разбивает по строкам, if x отфильтровывает пустые строки
    MAP = [list(x) for x in MAP.split("\n") if x]

    # Определение стоимостей перемещений
    cost_regular = 1.0  # Стоимость обычного перемещения (по вертикали/горизонтали)
    cost_diagonal = 1.7  # Стоимость диагонального перемещения (приблизительно √2)

    # Словарь стоимостей для всех возможных направлений движения
    COSTS = {
        "up": cost_regular,           # Вверх
        "down": cost_regular,         # Вниз
        "left": cost_regular,         # Влево
        "right": cost_regular,        # Вправо
        "up left": cost_diagonal,     # Вверх-влево (диагональ)
        "up right": cost_diagonal,    # Вверх-вправо (диагональ)
        "down left": cost_diagonal,   # Вниз-влево (диагональ)
        "down right": cost_diagonal,  # Вниз-вправо (диагональ)
    }

    # Создание экземпляра решателя лабиринта
    problem = MazeSolver(MAP)

    # Запуск алгоритма A* для поиска пути
    # graph_search=True означает, что алгоритм будет отслеживать посещенные состояния
    result = astar(problem, graph_search=True)

    # Извлечение пути из результата (список состояний, пройденных до цели)
    path = [x[1] for x in result.path()]

    # Вывод карты лабиринта с найденным путем
    print()
    for y in range(len(MAP)):
        for x in range(len(MAP[y])):
            if (x, y) == problem.initial:  # Начальная позиция
                print('o', end='')
            elif (x, y) == problem.goal:   # Целевая позиция
                print('x', end='')
            elif (x, y) in path:           # Ячейки пути
                print('·', end='')         # Символ точки для отображения пути
            else:                          # Остальные ячейки
                print(MAP[y][x], end='')   # Оригинальный символ карты
        print()  # Переход на новую строку после каждой строки карты