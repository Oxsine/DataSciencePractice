from sklearn.datasets import fetch_california_housing
from sklearn import datasets

print("Первый пример вывода:\n")

housing = fetch_california_housing()
print(housing.data[:5])  
print(housing.target[:5])

print("\n")

print("---" * 30)

print("\nПример второго вывода \n")

digits = datasets.load_digits()
print(digits.images[0])
