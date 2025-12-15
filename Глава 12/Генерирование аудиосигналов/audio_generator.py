import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

output_file = 'generated_audio.wav'

duration = 4
sampling_freq = 44100
tone_freq = 784 
min_val = -4 * np.pi
max_val = 4 * np.pi

t = np.linspace(min_val, max_val, duration * sampling_freq)
signal = np.sin(2 * np.pi * tone_freq * t)

noise = 0.5 * np.random.rand(duration * sampling_freq)
signal += noise

scaling_factor = np.power(2, 15) - 1
signal_normalized = signal / np.max(np.abs(signal))
signal_scaled = np.int16(signal_normalized * scaling_factor)

write(output_file, sampling_freq, signal_scaled)

signal = signal[:200]

time_axis = 1000 * np.arange(0, len(signal), 1) / float(sampling_freq) 

plt.plot(time_axis, signal, color='black')
plt.xlabel('Время (миллисекунды)')
plt.ylabel('Амплитуда')
plt.title('Сгенерированный аудиосигнал')
plt.show()