import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile

sampling_freq, signal = wavfile.read('random_sound.wav')

print('\nФорма сигнала:', signal.shape)
print('Тип данных:', signal.dtype)
print('Продолжительность сигнала:', round(signal.shape[0] / float(sampling_freq), 2), 'в секундах')

signal = signal / np.power(2, 15)

signal = signal[:50]

time_axis = 1000 * np.arange(0, len(signal), 1) / float(sampling_freq)

plt.plot(time_axis, signal, color='black')
plt.xlabel('Время (милисекунды)')
plt.ylabel('Аплетуда')
plt.title('Входной аудио сигнал')
plt.show()