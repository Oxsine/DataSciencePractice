import json
import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write

def tone_synthesizer(freq, duration, amplitude=1.0, sampling_freq=44100):
    num_samples = int(duration * sampling_freq)
    time_axis = np.linspace(0, duration, num_samples)

    signal = amplitude * np.sin(2 * np.pi * freq * time_axis)

    return signal.astype(np.int16) 

if __name__=='__main__':
    file_tone_single = 'generated_tone_single.wav'
    file_tone_sequence = 'generated_tone_sequence.wav'

    mapping_file = 'tone_mapping.json'

    try:
        with open(mapping_file, 'r') as f:
            tone_map = json.loads(f.read())
    except FileNotFoundError:
        print(f"Файл {mapping_file} не найден. Создаю его...")
        tone_map = {
            'C': 261.63,
            'D': 293.66,
            'E': 329.63,
            'F': 349.23,
            'G': 392.00, 
            'A': 440.00,
            'B': 493.88 
        }
        with open(mapping_file, 'w') as f:
            json.dump(tone_map, f)
        print(f"Файл {mapping_file} создан.")

    tone_name = 'F'
    duration = 3    
    amplitude = 12000
    sampling_freq = 44100 

    tone_freq = tone_map[tone_name]

    synthesized_tone = tone_synthesizer(tone_freq, duration, amplitude, sampling_freq)

    write(file_tone_single, sampling_freq, synthesized_tone)

    tone_sequence = [('G', 0.4), ('D', 0.5), ('F', 0.3), ('C', 0.6), ('A', 0.4)]

    signal = np.array([])
    for item in tone_sequence:

        tone_name = item[0]

        freq = tone_map[tone_name]

        note_duration = item[1]

        synthesized_tone = tone_synthesizer(freq, note_duration, amplitude, sampling_freq)

        signal = np.append(signal, synthesized_tone, axis=0)

    write(file_tone_sequence, sampling_freq, signal)
    
    print(f"Создан файл: {file_tone_single}")
    print(f"Создан файл: {file_tone_sequence}")