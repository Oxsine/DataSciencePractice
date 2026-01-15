import tensorflow as tf
import numpy as np

# Загружаем данные MNIST
mnist_data = tf.keras.datasets.mnist
(train_inputs, train_targets), (test_inputs, test_targets) = mnist_data.load_data()

# Подготовка данных
train_inputs = train_inputs.reshape(-1, 28, 28, 1).astype('float32') / 255.0
test_inputs = test_inputs.reshape(-1, 28, 28, 1).astype('float32') / 255.0
train_labels = tf.keras.utils.to_categorical(train_targets, 10)
test_labels = tf.keras.utils.to_categorical(test_targets, 10)

# Функции для инициализации параметров
def create_weights(shape):
    return tf.Variable(tf.random.truncated_normal(shape, stddev=0.1))

def create_bias(shape):
    return tf.Variable(tf.constant(0.1, shape=shape))

# Создание сверточной нейронной сети
class ConvolutionalDigitNet(tf.keras.Model):
    def __init__(self):
        super(ConvolutionalDigitNet, self).__init__()
        
        # Первый сверточный слой: 5x5 фильтр, 1 вход, 32 выхода
        self.conv1_kernel = create_weights([5, 5, 1, 32])
        self.conv1_offset = create_bias([32])
        
        # Второй сверточный слой: 5x5 фильтр, 32 входа, 64 выхода
        self.conv2_kernel = create_weights([5, 5, 32, 64])
        self.conv2_offset = create_bias([64])
        
        # Полносвязный скрытый слой
        self.fc1_matrix = create_weights([7 * 7 * 64, 1024])
        self.fc1_bias = create_bias([1024])
        
        # Выходной слой
        self.output_weights = create_weights([1024, 10])
        self.output_bias = create_bias([10])
        
        # Параметр dropout для регуляризации
        self.drop_prob = 0.5
    
    def call(self, inputs, is_training=False):
        # Первый сверточный блок: свертка + ReLU + пулинг
        conv1_out = tf.nn.relu(
            tf.nn.conv2d(inputs, self.conv1_kernel, strides=[1, 1, 1, 1], padding='SAME') + self.conv1_offset
        )
        pool1_out = tf.nn.max_pool2d(conv1_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        # Второй сверточный блок: свертка + ReLU + пулинг
        conv2_out = tf.nn.relu(
            tf.nn.conv2d(pool1_out, self.conv2_kernel, strides=[1, 1, 1, 1], padding='SAME') + self.conv2_offset
        )
        pool2_out = tf.nn.max_pool2d(conv2_out, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
        
        # Разворачиваем тензор для полносвязных слоев
        flattened = tf.reshape(pool2_out, [-1, 7 * 7 * 64])
        
        # Полносвязный слой с активацией ReLU
        fc1_out = tf.nn.relu(tf.matmul(flattened, self.fc1_matrix) + self.fc1_bias)
        
        # Применяем dropout только во время обучения
        if is_training:
            fc1_dropped = tf.nn.dropout(fc1_out, rate=self.drop_prob)
        else:
            fc1_dropped = fc1_out
        
        # Выходной слой (логиты)
        network_output = tf.matmul(fc1_dropped, self.output_weights) + self.output_bias
        
        return network_output

# Создаем экземпляр модели
conv_model = ConvolutionalDigitNet()

# Функция потерь и оптимизатор
loss_calculator = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
network_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)

# Функция для вычисления точности классификации
def calculate_accuracy(prediction_logits, ground_truth):
    predicted_indices = tf.argmax(prediction_logits, axis=1)
    true_indices = tf.argmax(ground_truth, axis=1)
    return tf.reduce_mean(tf.cast(tf.equal(predicted_indices, true_indices), tf.float32))

# Тренировочный цикл
print("Запускаем обучение сверточной сети...")
samples_per_batch = 75
total_training_steps = 5000

# Создаем tf.data.Dataset для обучения
training_data = tf.data.Dataset.from_tensor_slices((train_inputs, train_labels))
training_data = training_data.shuffle(buffer_size=10000).batch(samples_per_batch)

for step_count, (batch_input, batch_target) in enumerate(training_data.repeat()):
    if step_count >= total_training_steps:
        break
    
    with tf.GradientTape() as grad_tape:
        # Прямой проход с включенным dropout
        batch_predictions = conv_model(batch_input, is_training=True)
        step_loss = loss_calculator(batch_target, batch_predictions)
    
    # Вычисляем градиенты и обновляем параметры
    weight_gradients = grad_tape.gradient(step_loss, conv_model.trainable_variables)
    network_optimizer.apply_gradients(zip(weight_gradients, conv_model.trainable_variables))
    
    # Вывод прогресса каждые 50 шагов
    if step_count % 50 == 0:
        # Вычисляем точность без dropout
        batch_pred_no_dropout = conv_model(batch_input, is_training=False)
        batch_accuracy = calculate_accuracy(batch_pred_no_dropout, batch_target)
        print(f'Шаг {step_count}, Точность = {batch_accuracy.numpy():.4f}, Потери = {step_loss.numpy():.4f}')

# Финальная оценка на тестовой выборке
print("\nВычисляем финальную точность на тестовой выборке...")
final_predictions = conv_model(test_inputs, is_training=False)
final_accuracy = calculate_accuracy(final_predictions, test_labels)
print(f"Точность на тестовых данных: {final_accuracy.numpy():.4f}")