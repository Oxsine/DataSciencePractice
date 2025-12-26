import numpy as np
import matplotlib.pyplot as plt
from hmmlearn.hmm import GaussianHMM
import os

# Загрузка данных с пробелами как разделителями
file_name = 'data_2D.txt'

try:
    # Пробуем загрузить с пробелами как разделителями
    data = np.loadtxt(file_name, delimiter=None)  # None означает любой пробельный символ
    print(f"Данные загружены. Форма: {data.shape}")
    
except ValueError as e:
    print(f"Ошибка загрузки: {e}")
    print("\nПробуем другой подход...")
    
    # Читаем файл как текст
    with open(file_name, 'r') as f:
        lines = f.readlines()
    
    print(f"Первые 3 строки файла:")
    for i, line in enumerate(lines[:3]):
        print(f"Строка {i}: {line.strip()}")
    
    # Преобразуем строки в числа, игнорируя нечисловые значения
    data_list = []
    for line in lines:
        values = line.strip().split()
        numeric_values = []
        for val in values:
            try:
                numeric_values.append(float(val))
            except ValueError:
                pass  # Пропускаем нечисловые значения
        if numeric_values:  # Если есть числовые значения в строке
            data_list.append(numeric_values)
    
    data = np.array(data_list)
    print(f"Данные преобразованы. Форма: {data.shape}")

# Проверяем данные
print(f"\nПервые 5 строк данных:")
for i in range(min(5, len(data))):
    print(f"Строка {i}: {data[i]}")

# Извлечение данных для HMM
if data.ndim == 1:
    X = data.reshape(-1, 1)
elif data.shape[1] >= 3:
    X = data[:, 2].reshape(-1, 1)  # Третий столбец
else:
    X = data  # Используем все данные

print(f"\nДанные для HMM. Форма: {X.shape}")

# Создание и обучение модели HMM
num_components = 5
hmm = GaussianHMM(n_components=num_components,
                  covariance_type='diag', 
                  n_iter=1000)

try:
    hmm.fit(X)
    print("Модель успешно обучена!")
    
    # Вывод статистик
    print('\nСтатистика HMM:') 
    for i in range(hmm.n_components):
        print(f'\nСкрытое состояние {i+1}:')
        print(f'Среднее = {hmm.means_[i][0]:.2f}')
        print(f'Дисперсия = {np.diag(hmm.covars_[i])[0]:.2f}')
    
    # Генерация данных
    num_samples = 1200
    generated_data, _ = hmm.sample(num_samples)
    
    # Визуализация
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.plot(X[:500], c='blue', alpha=0.7, label='Исходные данные')
    plt.title('Исходные данные (первые 500 точек)')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    plt.plot(generated_data[:, 0], c="black", label='Сгенерированные данные')
    plt.title('Сгенерированные данные HMM')
    plt.xlabel('Время')
    plt.ylabel('Значение')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
except Exception as e:
    print(f"Ошибка при обучении модели: {e}")
