import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile 
from python_speech_features import mfcc, logfbank

sampling_freq, signal = wavfile.read('random_sound.wav')

signal = signal[:10000]

features_mfcc = mfcc(signal, sampling_freq)

print('\nMFCC:\nКоличество окон =', features_mfcc.shape[0])
print('Длина каждого признака =', features_mfcc.shape[1])

features_mfcc = features_mfcc.T
plt.matshow(features_mfcc)
plt.title('MFCC')

features_fb = logfbank(signal, sampling_freq)

print('\nФильтр-банк:\nКоличество окон =', features_fb.shape[0])
print('Длина каждого признака =', features_fb.shape[1])

features_fb = features_fb.T
plt.matshow(features_fb)
plt.title('Фильтр-банк')

plt.show()