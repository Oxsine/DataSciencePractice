import tensorflow as tf
import numpy as np

# Отключаем предупреждения TF 1.x
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

# Загружаем данные MNIST
mnist_dataset = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist_dataset.load_data()

# Преобразуем в float32 и нормализуем
train_images = train_images.reshape(-1, 784).astype('float32') / 255.0
test_images = test_images.reshape(-1, 784).astype('float32') / 255.0

# One-hot encoding для меток
train_labels_encoded = tf.keras.utils.to_categorical(train_labels, 10)
test_labels_encoded = tf.keras.utils.to_categorical(test_labels, 10)

# Параметры обучения
mini_batch_size = 90
training_steps = 1500
step_size = 0.5

# Создаем модель
class BasicMNISTClassifier(tf.keras.Model):
    def __init__(self):
        super(BasicMNISTClassifier, self).__init__()
        self.weight_matrix = tf.Variable(tf.zeros([784, 10]))
        self.bias_vector = tf.Variable(tf.zeros([10]))
    
    def call(self, inputs):
        return tf.matmul(inputs, self.weight_matrix) + self.bias_vector

# Создаем экземпляр модели
classifier = BasicMNISTClassifier()

# Функция потерь и оптимизатор
loss_function = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
gradient_optimizer = tf.keras.optimizers.SGD(learning_rate=step_size)

# Тренировочный цикл
print("Начинаем обучение...")
for step in range(training_steps):
    # Выбираем случайный батч
    batch_indices = np.random.choice(len(train_images), mini_batch_size, replace=False)
    input_batch = tf.convert_to_tensor(train_images[batch_indices])
    target_batch = tf.convert_to_tensor(train_labels_encoded[batch_indices])
    
    with tf.GradientTape() as gradient_tape:
        # Прямой проход
        model_output = classifier(input_batch)
        # Вычисляем потери
        current_loss = loss_function(target_batch, model_output)
    
    # Вычисляем градиенты и обновляем веса
    parameter_gradients = gradient_tape.gradient(current_loss, classifier.trainable_variables)
    gradient_optimizer.apply_gradients(zip(parameter_gradients, classifier.trainable_variables))
    
    # Выводим прогресс каждые 100 итераций
    if (step + 1) % 100 == 0:
        print(f"Шаг {step + 1}/{training_steps}, Потери: {current_loss.numpy():.4f}")

# Оценка точности
print("\nВычисляем точность на тестовых данных...")
test_outputs = classifier(tf.convert_to_tensor(test_images))
predicted_classes = tf.argmax(test_outputs, axis=1)
actual_classes = tf.argmax(test_labels_encoded, axis=1)

classification_accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted_classes, actual_classes), tf.float32))
print(f"Точность на тестовых данных: {classification_accuracy.numpy():.4f}")

# Примеры предсказаний
print("\nПримеры предсказаний для первых 5 тестовых изображений:")
for idx in range(5):
    predicted = predicted_classes[idx].numpy()
    actual = actual_classes[idx].numpy()
    print(f"Изображение {idx}: Предсказано {predicted}, Правильно {actual} {'✓' if predicted == actual else '✗'}")