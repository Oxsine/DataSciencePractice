import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.cluster import KMeans

# Загрузка данных из входного файла
X = np.loadtxt('../data/data_quality.txt', delimiter=',')

# Проверяем размерность данных
print(f"Размерность данных: {X.shape}")
print(f"Количество образцов: {X.shape[0]}")
print(f"Количество признаков: {X.shape[1]}")

# Инициализация переменных
scores = []
values = np.arange(2, 10)

# Итерирование в определенном диапазоне значений
for num_clusters in values:
    # Обучение модели кластеризации KMeans
    kmeans = KMeans(init='k-means++', n_clusters=num_clusters, n_init=10, random_state=42)
    kmeans.fit(X)
    
    # Получение силуэтной оценки
    score = metrics.silhouette_score(X, kmeans.labels_, metric='euclidean')
    
    # Вывод силуэтной оценки
    print(f"Количество кластеров = {num_clusters}")
    print(f"Силуэтная оценка = {score:.4f}")
    scores.append(score)

# Отображение силуэтных оценок на графике
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.bar(values, scores, width=0.7, color='blue', align='center')
plt.xlabel('Количество кластеров')
plt.ylabel('Силуэтная оценка')
plt.title('Зависимость силуэтной оценки от количества кластеров')
plt.grid(True, alpha=0.3)

# Добавление значений на столбцы
for i, v in enumerate(scores):
    plt.text(values[i], v + 0.01, f'{v:.3f}', ha='center', va='bottom')

# Извлечение наилучшей оценки и оптимального количества кластеров
optimal_idx = np.argmax(scores)
optimal_num_clusters = values[optimal_idx]
optimal_score = scores[optimal_idx]

print(f"\nОптимальное количество кластеров = {optimal_num_clusters}")
print(f"Максимальная силуэтная оценка = {optimal_score:.4f}")

# Обучение модели с оптимальным количеством кластеров
kmeans_optimal = KMeans(init='k-means++', n_clusters=optimal_num_clusters, n_init=10, random_state=42)
kmeans_optimal.fit(X)
labels = kmeans_optimal.labels_
centroids = kmeans_optimal.cluster_centers_

# Отображение данных на графике с кластерами
plt.subplot(1, 2, 2)
colors = ['red', 'green', 'blue', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']

# Отображаем точки данных
for i in range(optimal_num_clusters):
    cluster_points = X[labels == i]
    plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                color=colors[i % len(colors)], s=50, alpha=0.7,
                label=f'Кластер {i+1}')

# Отображаем центроиды
plt.scatter(centroids[:, 0], centroids[:, 1], 
            color='black', s=200, marker='X', 
            label='Центроиды', edgecolors='white', linewidths=2)

# Настройка границ графика
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

plt.title(f'Кластеризация данных (K={optimal_num_clusters})')
plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xlabel('Признак 1')
plt.ylabel('Признак 2')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# Дополнительная информация о кластерах
print("\nИнформация о кластерах:")
for i in range(optimal_num_clusters):
    cluster_size = np.sum(labels == i)
    cluster_ratio = cluster_size / len(X) * 100
    print(f"Кластер {i+1}: {cluster_size} точек ({cluster_ratio:.1f}%)")

# Отображение силуэтного графика для оптимального числа кластеров
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
silhouette_vals = metrics.silhouette_samples(X, labels)
y_lower = 10

for i in range(optimal_num_clusters):
    # Собираем силуэтные значения для i-го кластера и сортируем
    cluster_silhouette_vals = silhouette_vals[labels == i]
    cluster_silhouette_vals.sort()
    
    size_cluster_i = cluster_silhouette_vals.shape[0]
    y_upper = y_lower + size_cluster_i
    
    color = colors[i % len(colors)]
    ax.fill_betweenx(np.arange(y_lower, y_upper),
                     0, cluster_silhouette_vals,
                     facecolor=color, edgecolor=color, alpha=0.7)
    
    # Подписываем кластеры
    ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i+1))
    
    y_lower = y_upper + 10  # 10 для отступа между кластерами

ax.set_title(f"Силуэтный график для {optimal_num_clusters} кластеров")
ax.set_xlabel("Силуэтный коэффициент")
ax.set_ylabel("Кластер")
ax.axvline(x=optimal_score, color="red", linestyle="--", label=f"Средний: {optimal_score:.3f}")
ax.set_yticks([])  # Убираем метки на оси Y
ax.legend()

plt.show()