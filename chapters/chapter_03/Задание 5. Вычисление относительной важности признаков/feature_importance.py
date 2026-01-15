import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn import datasets
from sklearn.metrics import mean_squared_error, explained_variance_score
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

# Загрузка данных о жилье - используем калифорнийский набор данных вместо Boston
housing_data = datasets.fetch_california_housing()

# Перемешивание данных
X, y = shuffle(housing_data.data, housing_data.target, random_state=7)

# Разделение данных на обучающую и тестовую выборки 
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=7
)

# Модель регрессора AdaBoost
regressor = AdaBoostRegressor(
    DecisionTreeRegressor(max_depth=5),
    n_estimators=500,
    random_state=42
)
regressor.fit(X_train, y_train)

# Оценка производительности регрессора AdaBoost
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
evs = explained_variance_score(y_test, y_pred)
print("\nРЕГРЕССОР ADABOOST")
print("Среднеквадратичная ошибка =", round(mse, 3))
print("Объясненная дисперсия =", round(evs, 3))

# Извлечние важности признаков
feature_importances = regressor.feature_importances_
feature_names = housing_data.feature_names

# Преобразование feature_names в массив numpy строк для корректного индексирования
feature_names = np.array(feature_names)

# Нормализация значений важности
feature_importances = 100.0 * (feature_importances / max(feature_importances))

# Сортировка значений по убыванию
index_sorted = np.argsort(feature_importances)[::-1]  

# Расположение меток на оси X
pos = np.arange(len(feature_names)) + 0.5

# Построение столбчатой диаграммы
plt.figure(figsize=(12, 7))
plt.bar(pos, feature_importances[index_sorted], align='center')
plt.xticks(pos, feature_names[index_sorted], rotation=45, ha='right')
plt.ylabel('Относительная важность')
plt.title('Важность признаков - AdaBoost Регрессор')
plt.tight_layout()
plt.show()