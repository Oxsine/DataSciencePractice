import json  # Импорт модуля для работы с JSON файлами (чтение/запись)
import numpy as np  # Импорт библиотеки для работы с массивами и математическими операциями
import matplotlib.pyplot as plt  # Импорт библиотеки для построения графиков
from scipy.io.wavfile import write  # Импорт функции записи WAV файлов

def tone_synthesizer(freq, duration, amplitude=1.0, sampling_freq=44100):
    """
    Синтезатор тона: генерирует синусоидальный сигнал заданной частоты.
    
    Args:
        freq: Частота тона в Герцах (Гц)
        duration: Длительность тона в секундах
        amplitude: Амплитуда сигнала (по умолчанию 1.0)
        sampling_freq: Частота дискретизации (по умолчанию 44100 Гц)
        
    Returns:
        Аудиосигнал в формате 16-битного целого числа
    """
    # Вычисление количества отсчетов = длительность × частота дискретизации
    num_samples = int(duration * sampling_freq)
    
    # Создание временной оси от 0 до duration с равномерным шагом
    time_axis = np.linspace(0, duration, num_samples)

    # Генерация синусоидального сигнала: A × sin(2π × f × t)
    signal = amplitude * np.sin(2 * np.pi * freq * time_axis)

    # Преобразование сигнала в 16-битный целочисленный формат для записи в WAV
    return signal.astype(np.int16) 

if __name__=='__main__':
    # Имена выходных файлов для одиночного тона и последовательности тонов
    file_tone_single = 'generated_tone_single.wav'
    file_tone_sequence = 'generated_tone_sequence.wav'

    # Имя JSON файла с соответствием нот и частот
    mapping_file = 'tone_mapping.json'

    # Загрузка таблицы соответствия нот и частот из JSON файла
    try:
        with open(mapping_file, 'r') as f:
            tone_map = json.loads(f.read())  # Чтение и парсинг JSON
    except FileNotFoundError:
        # Если файл не найден, создаем стандартное соответствие нот и частот
        print(f"Файл {mapping_file} не найден. Создаю его...")
        # Частоты для нот первой октавы (в Герцах):
        tone_map = {
            'C': 261.63,  # До
            'D': 293.66,  # Ре
            'E': 329.63,  # Ми
            'F': 349.23,  # Фа
            'G': 392.00,  # Соль
            'A': 440.00,  # Ля
            'B': 493.88   # Си
        }
        # Сохранение таблицы в JSON файл для последующего использования
        with open(mapping_file, 'w') as f:
            json.dump(tone_map, f)
        print(f"Файл {mapping_file} создан.")

    # Параметры для генерации одиночного тона
    tone_name = 'F'  # Нота Фа
    duration = 3     # Длительность 3 секунды
    amplitude = 12000  # Амплитуда (максимальное значение для 16-битного аудио ≈ 32768)
    sampling_freq = 44100  # Стандартная частота дискретизации для аудио

    # Получение частоты для выбранной ноты из таблицы соответствия
    tone_freq = tone_map[tone_name]

    # Генерация одиночного тона с помощью функции-синтезатора
    synthesized_tone = tone_synthesizer(tone_freq, duration, amplitude, sampling_freq)

    # Запись одиночного тона в WAV файл
    write(file_tone_single, sampling_freq, synthesized_tone)

    # Создание последовательности тонов: список кортежей (нота, длительность)
    tone_sequence = [('G', 0.4), ('D', 0.5), ('F', 0.3), ('C', 0.6), ('A', 0.4)]

    # Инициализация пустого массива для результирующего сигнала
    signal = np.array([])
    
    # Генерация и объединение всех тонов из последовательности
    for item in tone_sequence:
        tone_name = item[0]  # Получение названия ноты
        freq = tone_map[tone_name]  # Получение частоты ноты из таблицы
        note_duration = item[1]  # Получение длительности ноты

        # Генерация тона для текущей ноты
        synthesized_tone = tone_synthesizer(freq, note_duration, amplitude, sampling_freq)

        # Добавление сгенерированного тона к общему сигналу
        signal = np.append(signal, synthesized_tone, axis=0)

    # Запись последовательности тонов в WAV файл
    write(file_tone_sequence, sampling_freq, signal)
    
    # Вывод информации о созданных файлах
    print(f"Создан файл: {file_tone_single}")
    print(f"Создан файл: {file_tone_sequence}")