import datetime
import warnings
import numpy as np
import matplotlib.pyplot as plt 
import yfinance as yf
from hmmlearn.hmm import GaussianHMM

# Загрузка исторических биржевых котировок через yfinance
start = datetime.date(1970, 9, 4)
end = datetime.date(2016, 5, 17)

print("Загрузка данных...")
# Скачиваем данные
stock_data = yf.download('INTC', start=start, end=end)

print(f"Загружено {len(stock_data)} записей")
print(f"Колонки: {stock_data.columns.tolist()}")

# Извлечение ежедневных котировок на момент закрытия биржи
closing_quotes = stock_data['Close'].values
print(f"Закрытие: {closing_quotes[:5]}...")

# Извлечение ежедневных объемов проторгованных акций
volumes = stock_data['Volume'].values[1:]
print(f"Объемы: {volumes[:5]}...")

# Проверка на NaN значения
print(f"NaN в closing_quotes: {np.isnan(closing_quotes).sum()}")
print(f"NaN в volumes: {np.isnan(volumes).sum()}")

# Очистка от NaN значений
closing_quotes_clean = closing_quotes[~np.isnan(closing_quotes)]
volumes_clean = volumes[~np.isnan(volumes)]

# Проверка размеров после очистки
min_len = min(len(closing_quotes_clean), len(volumes_clean))
closing_quotes_clean = closing_quotes_clean[:min_len]
volumes_clean = volumes_clean[:min_len]

# Определение процентной разницы котировок на момент закрытия биржи
# Безопасное вычисление с проверкой на ноль
closing_quotes_clean = np.array(closing_quotes_clean, dtype=np.float64)
# Добавляем небольшое значение чтобы избежать деления на ноль
closing_quotes_clean = np.where(closing_quotes_clean == 0, 1e-10, closing_quotes_clean)

diff_percentages = 100.0 * np.diff(closing_quotes_clean) / closing_quotes_clean[:-1]

print(f"Размер diff_percentages: {len(diff_percentages)}")
print(f"Размер volumes_clean: {len(volumes_clean)}")
print(f"Первые 5 значений diff_percentages: {diff_percentages[:5]}")

# Выравниваем размеры массивов
min_len = min(len(diff_percentages), len(volumes_clean))
diff_percentages = diff_percentages[:min_len]
volumes_clean = volumes_clean[:min_len]

# Попарная упаковка разностей и объемов акций для тренировки
training_data = np.column_stack((diff_percentages, volumes_clean))
print(f"Размер training_data: {training_data.shape}")

# Проверка на NaN в тренировочных данных
nan_mask = np.isnan(training_data).any(axis=1)
if nan_mask.any():
    print(f"Найдены NaN в training_data: {nan_mask.sum()}")
    training_data = training_data[~nan_mask]

# Создадим и обучим гауссовскую модель HMM
print("Обучение HMM модели...")
hmm = GaussianHMM(n_components=7, covariance_type='diag', n_iter=1000, random_state=42)
with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    hmm.fit(training_data)

print("Обучение завершено!")

# Генерирование данных с использованием HMM-модели
num_samples = 300
samples, _ = hmm.sample(num_samples)

# Построение графика процентных разниц
plt.figure(figsize=(14, 10))

# График 1: Сгенерированные процентные изменения
plt.subplot(3, 2, 1)
plt.title('Сгенерированные процентные изменения (HMM)')
plt.plot(np.arange(num_samples), samples[:, 0], c='red', linewidth=1)
plt.xlabel('Дни')
plt.ylabel('Процентное изменение, %')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# График 2: Сгенерированные объемы
plt.subplot(3, 2, 2)
plt.title('Сгенерированные объемы (HMM)')
plt.plot(np.arange(num_samples), samples[:, 1], c='red', linewidth=1)
plt.ylim(ymin=0)
plt.xlabel('Дни')
plt.ylabel('Объем')
plt.grid(True, alpha=0.3)

# График 3: Реальные процентные изменения
plt.subplot(3, 2, 3)
plt.title('Реальные процентные изменения (INTC)')
plt.plot(np.arange(min_len)[:300], diff_percentages[:300], c='blue', linewidth=1)
plt.xlabel('Дни')
plt.ylabel('Процентное изменение, %')
plt.grid(True, alpha=0.3)
plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5)

# График 4: Реальные объемы
plt.subplot(3, 2, 4)
plt.title('Реальные объемы (INTC)')
plt.plot(np.arange(min_len)[:300], volumes_clean[:300], c='blue', linewidth=1)
plt.ylim(ymin=0)
plt.xlabel('Дни')
plt.ylabel('Объем')
plt.grid(True, alpha=0.3)

# График 5: Сравнение распределений процентных изменений
plt.subplot(3, 2, 5)
plt.title('Распределение процентных изменений')
plt.hist(samples[:, 0], bins=50, alpha=0.5, color='red', label='HMM', density=True)
plt.hist(diff_percentages[:300], bins=50, alpha=0.5, color='blue', label='Реальные', density=True)
plt.xlabel('Процентное изменение, %')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True, alpha=0.3)

# График 6: Сравнение распределений объемов
plt.subplot(3, 2, 6)
plt.title('Распределение объемов')
plt.hist(samples[:, 1], bins=50, alpha=0.5, color='red', label='HMM', density=True)
plt.hist(volumes_clean[:300], bins=50, alpha=0.5, color='blue', label='Реальные', density=True)
plt.xlabel('Объем')
plt.ylabel('Плотность')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Вывод статистики
print("\n=== Статистика ===")
print(f"Реальные процентные изменения:")
print(f"  Среднее: {np.mean(diff_percentages):.4f}%")
print(f"  Стандартное отклонение: {np.std(diff_percentages):.4f}%")
print(f"  Минимум: {np.min(diff_percentages):.4f}%")
print(f"  Максимум: {np.max(diff_percentages):.4f}%")

print(f"\nСгенерированные процентные изменения:")
print(f"  Среднее: {np.mean(samples[:, 0]):.4f}%")
print(f"  Стандартное отклонение: {np.std(samples[:, 0]):.4f}%")
print(f"  Минимум: {np.min(samples[:, 0]):.4f}%")
print(f"  Максимум: {np.max(samples[:, 0]):.4f}%")

print(f"\nРеальные объемы:")
print(f"  Среднее: {np.mean(volumes_clean):.0f}")
print(f"  Стандартное отклонение: {np.std(volumes_clean):.0f}")

print(f"\nСгенерированные объемы:")
print(f"  Среднее: {np.mean(samples[:, 1]):.0f}")
print(f"  Стандартное отклонение: {np.std(samples[:, 1]):.0f}")
