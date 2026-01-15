import os 
import argparse 
import string 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.model_selection import train_test_split
import sklearn_crfsuite
from sklearn_crfsuite import metrics
import warnings
warnings.filterwarnings('ignore')

# Генерация синтетических данных (аналог letters dataset)
def generate_letter_data(n_samples=1000, seq_length=10):
    """
    Генерация синтетических данных для распознавания букв
    В реальном применении загружайте свои данные
    """
    np.random.seed(42)
    alphabets = list(string.ascii_lowercase)
    n_letters = len(alphabets)
    
    X = []
    y = []
    
    for _ in range(n_samples):
        # Генерируем последовательность букв
        seq_len = np.random.randint(5, seq_length)
        letters_indices = np.random.randint(0, n_letters, seq_len)
        
        # Создаем признаки для каждой буквы в последовательности
        sequence_features = []
        for idx in letters_indices:
            # Простые признаки для демонстрации
            features = {
                'letter_index': idx,
                'is_vowel': idx in [0, 4, 8, 14, 20],  # a, e, i, o, u
                'position_in_seq': idx % 3,
                'prev_letter_diff': np.random.randint(-5, 6)
            }
            sequence_features.append(features)
        
        X.append(sequence_features)
        y.append([str(i) for i in letters_indices])  # метки как строки
    
    return X, y, alphabets

def build_arg_parser():
    parser = argparse.ArgumentParser(description="Trains a Conditional Random Field classifier")
    parser.add_argument('--C', dest='c_val', required=False, 
                       type=float, default=1.0, help='C value to be used for training')
    return parser

# Класс, моделирующий CRF с использованием sklearn-crfsuite
class CRFModel(object):
    def __init__(self, c_val=1.0):
        self.clf = sklearn_crfsuite.CRF(
            algorithm='lbfgs',
            c1=c_val,  # L1 regularization
            c2=0.1,    # L2 regularization
            max_iterations=100,
            all_possible_transitions=True
        )
        self.label_encoder = {}  # для преобразования меток
    
    # Загрузка/генерация тренировочных данных
    def load_data(self):
        X, y, alphabets = generate_letter_data(n_samples=2000)
        
        # Создаем простые "folds" для разделения на train/test
        folds = np.zeros(len(X))
        folds[:int(0.7 * len(X))] = 1  # 70% тренировочных
        
        return X, y, folds
    
    # Тренировка CRF
    def train(self, X_train, y_train):
        self.clf.fit(X_train, y_train)
    
    # Вычисление точности модели CRF
    def evaluate(self, X_test, y_test):
        y_pred = self.clf.predict(X_test)
        
        # Вычисляем точность по последовательностям
        correct = 0
        total = len(y_test)
        for true_seq, pred_seq in zip(y_test, y_pred):
            if true_seq == pred_seq:
                correct += 1
        
        return correct / total
    
    # Выполнение CRF для неизвестных данных
    def classify(self, input_data):
        return self.clf.predict([input_data])[0]

# Преобразование индексов в буквы
def convert_to_letters(indices, alphabets=None):
    if alphabets is None:
        alphabets = list(string.ascii_lowercase)
    
    # Если indices - список строк (меток), преобразуем в индексы
    if isinstance(indices[0], str):
        indices = [int(idx) for idx in indices]
    
    # Извлечение букв на основании индексов
    output = ''.join([alphabets[idx % len(alphabets)] for idx in indices])
    return output

if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    c_val = args.c_val
    
    # Создание CRF-модели
    crf = CRFModel(c_val)
    
    # Загрузка тренировочных и тестовых данных
    X, y, alphabets = generate_letter_data(n_samples=2000)
    
    # Разделение данных на тренировочные и тестовые
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Обучение CRF-модели
    print('\nTraining the CRF model...')
    crf.train(X_train, y_train)
    
    # Вычисление точности
    score = crf.evaluate(X_test, y_test)
    print(f'\nAccuracy score = {score*100:.2f}%')
    
    # Демонстрация предсказаний
    print("\n--- Примеры предсказаний ---")
    test_indices = range(0, min(5, len(X_test)))
    for idx in test_indices:
        print(f"\nПример {idx + 1}:")
        print(f"Исходная последовательность = {convert_to_letters(y_test[idx], alphabets)}")
        predicted = crf.classify(X_test[idx])
        print(f"Предсказанная последовательность = {convert_to_letters(predicted, alphabets)}")
