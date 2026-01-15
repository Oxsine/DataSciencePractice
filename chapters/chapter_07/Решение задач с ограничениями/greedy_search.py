import argparse  # Импорт модуля для обработки аргументов командной строки
import simpleai.search as ss  # Импорт библиотеки для поисковых алгоритмов

def build_arg_parser():
    """Создает и настраивает парсер аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Создаёт целевую строку с помощью жадного алгоритма')
    # Обязательный аргумент: целевая строка, которую нужно получить
    parser.add_argument("--input-string", dest="input_string", required=True,
            help="Целевая строка")
    # Необязательный аргумент: начальное состояние (по умолчанию - пустая строка)
    parser.add_argument("--initial-state", dest="initial_state", required=False,
            default='', help="Начальное состояние для поиска")
    return parser

class CustomProblem(ss.SearchProblem):
    """Пользовательский класс задачи поиска для построения целевой строки"""
    
    def set_target(self, target_string):
        """Устанавливает целевую строку для задачи"""
        self.target_string = target_string

    def actions(self, cur_state):
        """
        Возвращает список возможных действий из текущего состояния.
        Действие - это добавление одного символа.
        """
        # Если текущая строка короче целевой, можно добавлять символы
        if len(cur_state) < len(self.target_string):
            # Все буквы русского алфавита в нижнем и верхнем регистре, а также пробел
            alphabets = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
            return list(alphabets + ' ' + alphabets.upper())
        else:
            # Если достигнута длина целевой строки, больше действий нет
            return []

    def result(self, cur_state, action):
        """
        Применяет действие (добавляет символ) к текущему состоянию
        и возвращает новое состояние.
        """
        return cur_state + action

    def is_goal(self, cur_state):
        """Проверяет, является ли текущее состояние целевым"""
        return cur_state == self.target_string

    def heuristic(self, cur_state):
        """
        Эвристическая функция для оценки расстояния от текущего состояния до цели.
        Используется в жадном алгоритме для выбора следующего состояния.
        """
        # Вычисляем количество несовпадающих символов на соответствующих позициях
        # (только для той части, где строки перекрываются)
        dist = sum([1 if cur_state[i] != self.target_string[i] else 0
                    for i in range(len(cur_state))])

        # Разница в длине между целевой и текущей строкой
        diff = len(self.target_string) - len(cur_state)

        # Общая оценка: сумма несовпадений и недостающих символов
        return dist + diff 

if __name__ == '__main__':
    # Парсинг аргументов командной строки
    args = build_arg_parser().parse_args()

    # Создание экземпляра задачи
    problem = CustomProblem()

    # Настройка задачи: установка целевой строки и начального состояния
    problem.set_target(args.input_string)
    problem.initial_state = args.initial_state

    # Запуск жадного поиска (greedy search) для решения задачи
    output = ss.greedy(problem)

    # Вывод результатов
    print('\nЦелевая строка:', args.input_string)
    print('\nПуть к решению:')
    
    # Вывод каждого шага в пути решения
    for item in output.path():
        print(item)