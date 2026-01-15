import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

print("Версия TensorFlow:", tf.__version__)

# Определяем количество точек для генерации
points_count = 1500

# Генерируем данные на основе уравнения y = kx + b
dataset = []
k = 0.2
b_true = 0.5
for idx in range(points_count):
    # Генерируем значение 'x'
    x_val = np.random.normal(0.0, 0.8)
    
    # Добавляем случайный шум
    noise_val = np.random.normal(0.0, 0.04)
    
    # Вычисляем значение 'y'
    y_val = k * x_val + b_true + noise_val
    
    dataset.append([x_val, y_val])

# Разделяем данные на x и y
x_vals = np.array([item[0] for item in dataset])
y_vals = np.array([item[1] for item in dataset])

# Отображаем сгенерированные данные
plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_vals, 'ro', alpha=0.5)
plt.title('Исходные данные')
plt.xlabel('X')
plt.ylabel('Y')
plt.show()

# Инициализируем веса и смещение
weight = tf.Variable(tf.random.uniform([1], -1.0, 1.0))
bias = tf.Variable(tf.zeros([1]))

# Создаем оптимизатор
opt = tf.keras.optimizers.SGD(learning_rate=0.5)

# Цикл обучения
epochs = 10
loss_log = []

for epoch in range(epochs):
    with tf.GradientTape() as tape:
        # Вычисляем предсказанные значения y
        y_predicted = weight * x_vals + bias
        # Вычисляем функцию потерь
        current_loss = tf.reduce_mean(tf.square(y_predicted - y_vals))
    
    # Вычисляем градиенты
    grads = tape.gradient(current_loss, [weight, bias])
    
    # Обновляем параметры
    opt.apply_gradients(zip(grads, [weight, bias]))
    
    # Сохраняем значение потерь
    loss_log.append(current_loss.numpy())
    
    # Выводим информацию о текущей итерации
    print('\nЭПОХА', epoch + 1)
    print('Вес =', weight.numpy()[0])
    print('Смещение =', bias.numpy()[0])
    print('Потери =', current_loss.numpy())

    # Отображаем исходные данные и линию регрессии
    plt.figure(figsize=(10, 6))
    plt.plot(x_vals, y_vals, 'ro', alpha=0.5, label='Исходные точки')
    plt.plot(x_vals, weight.numpy() * x_vals + bias.numpy(), 
             'b-', linewidth=2, label=f'Предсказание (эпоха {epoch + 1})')
    
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Эпоха {epoch + 1} из {epochs}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# Финальные результаты
print("\n" + "="*50)
print("ИТОГОВЫЕ РЕЗУЛЬТАТЫ:")
print(f"Истинный коэффициент k: {k}")
print(f"Найденный вес: {weight.numpy()[0]:.4f}")
print(f"Истинное смещение b: {b_true}")
print(f"Найденное смещение: {bias.numpy()[0]:.4f}")
print("="*50)

# График изменения потерь
plt.figure(figsize=(10, 6))
plt.plot(range(1, epochs + 1), loss_log, 'bo-', linewidth=2)
plt.xlabel('Эпоха')
plt.ylabel('Потери')
plt.title('Динамика потерь в процессе обучения')
plt.grid(True, alpha=0.3)
plt.show()