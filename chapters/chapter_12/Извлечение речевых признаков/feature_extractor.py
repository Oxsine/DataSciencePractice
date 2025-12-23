import numpy as np  # Импорт библиотеки для работы с массивами и математическими операциями
import matplotlib.pyplot as plt  # Импорт библиотеки для построения графиков
from scipy.io import wavfile  # Импорт функции для чтения WAV файлов
from python_speech_features import mfcc, logfbank  # Импорт функций для извлечения аудио-признаков

# Чтение WAV файла: функция возвращает частоту дискретизации и аудиосигнал
sampling_freq, signal = wavfile.read('random_sound.wav')

# Ограничиваем сигнал первыми 10000 отсчетами для анализа (примерно 0.23 сек при 44.1 кГц)
# Это ускоряет обработку и визуализацию
signal = signal[:10000]

# Извлечение MFCC (Mel-frequency cepstral coefficients - мел-кепстральные коэффициенты)
# MFCC - популярные признаки для распознавания речи, моделирующие восприятие звука человеком
features_mfcc = mfcc(signal, sampling_freq)

# Вывод информации о MFCC признаках
print('\nMFCC:\nКоличество окон =', features_mfcc.shape[0])  # Количество временных окон
print('Длина каждого признака =', features_mfcc.shape[1])  # Количество MFCC коэффициентов (обычно 13)

# Транспонирование матрицы MFCC для визуализации (временная ось по горизонтали, коэффициенты по вертикали)
features_mfcc = features_mfcc.T
plt.matshow(features_mfcc)  # Построение тепловой карты MFCC
plt.title('MFCC')  # Заголовок графика

# Извлечение признаков фильтр-банка (log mel-filter bank energies)
# Логарифмическая энергия полос мел-фильтров - более простые признаки, чем MFCC
features_fb = logfbank(signal, sampling_freq)

# Вывод информации о признаках фильтр-банка
print('\nФильтр-банк:\nКоличество окон =', features_fb.shape[0])  # Количество временных окон
print('Длина каждого признака =', features_fb.shape[1])  # Количество фильтров в банке (обычно 26)

# Транспонирование матрицы фильтр-банка для визуализации
features_fb = features_fb.T
plt.matshow(features_fb)  # Построение тепловой карты фильтр-банка
plt.title('Фильтр-банк')  # Заголовок графика

plt.show()  # Отображение обоих графиков