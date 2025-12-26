from sklearn.datasets import make_classification
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.pipeline import Pipeline
from sklearn.ensemble import ExtraTreesClassifier

#Генерирование данных
X, у= make_classification(n_samples=150,
n_features=25, n_classes=3,
n_informative=6, n_redundant=0, random_state=7)

#Выбор k наиболее важных признаков
k_best_selector = SelectKBest(f_regression, k=9)

#Инициализация классификатора на основе предельно случайного леса
classifier = ExtraTreesClassifier(n_estimators=60, max_depth=4)

#Создание конвейера
processor_pipeline = Pipeline([('selector', k_best_selector), ('erf', classifier)])

#Установка параметров
processor_pipeline.set_params(selector__k=7, erf__n_estimators=30)

#Обучение конвейера
processor_pipeline.fit(X, у)

#Прогнозирование результатов для входных данных
output = processor_pipeline.predict(X)
print("\nПредсказанный результат:\n", output)

#Вывод оценки
print("\nОценка:", processor_pipeline.score(X, у))

#Вывод признаков, отобранных селектором конвейера
status = processor_pipeline.named_steps['selector'] .get_support ()

#Извлечение и вывод индексов выбранных признаков
selected = [i for i, X in enumerate(status) if X]
print("\nИндексы выбранных признаков:", ', '.join( [str(X) for i in selected]))
