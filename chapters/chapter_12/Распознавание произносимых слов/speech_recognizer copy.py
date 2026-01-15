import os  # Импорт модуля для работы с файловой системой
import argparse  # Импорт модуля для обработки аргументов командной строки
import warnings  # Импорт модуля для управления предупреждениями

import numpy as np  # Импорт библиотеки для работы с массивами и математических операций
from scipy.io import wavfile  # Импорт функции для чтения WAV файлов

from hmmlearn import hmm  # Импорт скрытой марковской модели из библиотеки hmmlearn
from python_speech_features import mfcc  # Импорт функции для извлечения MFCC признаков

def build_arg_parser():
    """Создает и настраивает парсер аргументов командной строки"""
    parser = argparse.ArgumentParser(description='Обучает систему распознавания речи на основе HMM')
    # Обязательный аргумент: папка с аудиофайлами для обучения
    parser.add_argument("--input-folder", dest="input_folder", required=True,
            help="Входная папка с аудиофайлами для обучения")
    return parser

class ModelHMM(object):
    """Класс для работы со скрытой марковской моделью (HMM)"""
    
    def __init__(self, num_components=4, num_iter=1000):
        """
        Инициализация HMM модели.
        
        Args:
            num_components: Количество состояний в HMM (по умолчанию 4)
            num_iter: Количество итераций обучения (по умолчанию 1000)
        """
        self.n_components = num_components
        self.n_iter = num_iter

        self.cov_type = 'diag'  # Тип ковариационной матрицы ('diag' - диагональная)
        self.model_name = 'GaussianHMM'  # Имя модели

        self.models = []  # Список для хранения обученных моделей

        # Создание GaussianHMM модели с заданными параметрами
        self.model = hmm.GaussianHMM(n_components=self.n_components, 
                covariance_type=self.cov_type, n_iter=self.n_iter)

    def train(self, training_data):
        """Обучение HMM модели на предоставленных данных"""
        np.seterr(all='ignore')  # Игнорирование всех ошибок вычислений для стабильности
        cur_model = self.model.fit(training_data)  # Обучение модели на тренировочных данных
        self.models.append(cur_model)  # Добавление обученной модели в список

    def compute_score(self, input_data):
        """Вычисление логарифмической вероятности данных при данной модели"""
        return self.model.score(input_data)  # Возвращает log-вероятность последовательности

def build_models(input_folder):
    """
    Построение HMM моделей для каждого класса (слова) из аудиофайлов в папке.
    
    Args:
        input_folder: Путь к папке с подпапками для каждого класса
        
    Returns:
        Список кортежей (обученная модель, метка класса)
    """
    speech_models = []  # Список для хранения всех обученных моделей

    # Проход по всем элементам во входной папке
    for dirname in os.listdir(input_folder):
        subfolder = os.path.join(input_folder, dirname)  # Полный путь к подпапке

        if not os.path.isdir(subfolder):  # Пропускаем файлы (работаем только с папками)
            continue

        # Извлекаем метку класса из имени подпапки (последняя часть пути)
        label = subfolder[subfolder.rfind('/') + 1:]

        X = np.array([])  # Инициализация массива для хранения всех MFCC признаков

        # Получение списка тренировочных файлов (все wav файлы, кроме последнего)
        training_files = [x for x in os.listdir(subfolder) if x.endswith('.wav')][:-1]

        # Обработка каждого тренировочного файла
        for filename in training_files: 
            filepath = os.path.join(subfolder, filename)  # Полный путь к файлу

            # Чтение аудиофайла
            sampling_freq, signal = wavfile.read(filepath)

            # Извлечение MFCC признаков с подавлением предупреждений
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                features_mfcc = mfcc(signal, sampling_freq)

            # Объединение признаков из всех файлов в один массив
            if len(X) == 0:
                X = features_mfcc
            else:
                X = np.append(X, features_mfcc, axis=0)

        # Создание и обучение HMM модели для текущего класса
        model = ModelHMM()
        model.train(X)
        speech_models.append((model, label))  # Сохранение модели с меткой

        model = None  # Освобождение ссылки на модель

    return speech_models

def run_tests(test_files):
    """
    Тестирование обученных моделей на тестовых файлах.
    
    Args:
        test_files: Список путей к тестовым аудиофайлам
    """
    for test_file in test_files:
        # Чтение тестового аудиофайла
        sampling_freq, signal = wavfile.read(test_file)

        # Извлечение MFCC признаков из тестового файла
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            features_mfcc = mfcc(signal, sampling_freq)

        max_score = -float('inf')  # Инициализация максимальной вероятности
        output_label = None  # Метка с максимальной вероятностью
        predicted_label = None  # Предсказанная метка

        # Вычисление вероятности для каждой модели
        for item in speech_models:
            model, label = item
            score = model.compute_score(features_mfcc)  # Вычисление log-вероятности
            if score > max_score:
                max_score = score
                predicted_label = label  # Обновление предсказанной метки

        # Извлечение оригинальной метки из пути файла
        start_index = test_file.find('/') + 1
        end_index = test_file.rfind('/')
        original_label = test_file[start_index:end_index]
        
        # Вывод результатов
        print('\nОригинал: ', original_label + "v")  # Оригинальная метка
        print('Предсказано:', predicted_label)  # Предсказанная моделью метка

if __name__ == '__main__':
    # Парсинг аргументов командной строки
    args = build_arg_parser().parse_args()
    input_folder = args.input_folder

    # Построение моделей HMM для всех классов во входной папке
    speech_models = build_models(input_folder)

    # Сбор тестовых файлов (файлы, содержащие '15' в имени)
    test_files = []
    for root, dirs, files in os.walk(input_folder):
        for filename in (x for x in files if '15' in x):
            filepath = os.path.join(root, filename)
            test_files.append(filepath)

    # Запуск тестирования на собранных тестовых файлах
    run_tests(test_files)