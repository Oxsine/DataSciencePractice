import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, mean_absolute_error
from sklearn.model_selection import train_test_split 
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesRegressor

# Загрузка входных данных
input_file = 'traffic_data.txt'
data = []
with open(input_file, 'r') as f:
    for line in f.readlines():
        items = line[:-1].split(',')
        data.append(items)

data = np.array(data)

# Преобразование строковых данных в числовые 
label_encoder = [] 
X_encoded = np.empty(data.shape)
for i, item in enumerate(data[0]):
    if item.isdigit():
        X_encoded[:, i] = data[:, i]
    else:
        label_encoder.append(preprocessing.LabelEncoder())
        X_encoded[:, i] = label_encoder[-1].fit_transform(data[:, i])

X = X_encoded[:, :-1].astype(int)
y = X_encoded[:, -1].astype(int)

# Разбиение данных на обучающий и тестовый наборы 
X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=5)

# Регрессор на основе метода "Чрезвычайно случайных лесов"
params = {'n_estimators': 100, 'max_depth': 6, 'random_state': 42}
regressor = ExtraTreesRegressor(**params)
regressor.fit(X_train, y_train)

# Вычисление характеристик эффективности регрессора на тестовых данных
y_pred = regressor.predict(X_test)
print("=== РЕЗУЛЬТАТЫ ПРОГНОЗИРОВАНИЯ ИНТЕНСИВНОСТИ ТРАФИКА ===")
print(f"Средняя абсолютная ошибка: {mean_absolute_error(y_test, y_pred):.3f}")

# Тестирование кодирования на отдельном примере данных 
test_datapoint = ['Monday', '08:30', 'Chicago', 'yes']
test_datapoint_encoded = [-1] * len(test_datapoint)
count = 0
for i, item in enumerate(test_datapoint):
    if item.isdigit():
        test_datapoint_encoded[i] = int(test_datapoint[i])
    else:
        try:
            test_datapoint_encoded[i] = int(label_encoder[count].transform([test_datapoint[i]])[0])
        except ValueError:
            label_encoder[count].classes_ = np.append(label_encoder[count].classes_, test_datapoint[i])
            test_datapoint_encoded[i] = int(label_encoder[count].transform([test_datapoint[i]])[0])
        count = count + 1

test_datapoint_encoded = np.array(test_datapoint_encoded)

# Прогнозирование результата для тестовой точки данных
print("Предсказанная интенсивность трафика:", int(regressor.predict([test_datapoint_encoded])[0]))

