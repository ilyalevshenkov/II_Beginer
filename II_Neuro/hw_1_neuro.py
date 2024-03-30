'''
Урок 1. Основы обучения нейронных сетей Попробуйте видоизменить параметры разобранной на уроке нейронной сети таким образом, чтобы улучшить её точность. Проведите анализ:
Что приводит к ухудшению точности нейронной сети?
Что приводит к увеличению её точности?

Построение двухслойной нейронный сети для классификации цветков ириса
Характеристики имеющейся нейронной сети:
1. Количество слоев: 2 (один скрытый слой).
2. Количество нейронов во входном слое: 4.
3. Количество нейронов в скрытом слое: 5.
4. Количество нейронов в выходном слое: 3.
5. Функция активации: сигмоида для всех нейронов.
6. Скорость обучения (learning rate): 0.001.
7. Количество итераций обучения: 5,000,000.
8. Метод обучения: обратное распространение (backpropagation) с использованием градиентного спуска.
'''

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# sklearn здесь только, чтобы разделить выборку на тренировочную и тестовую
from sklearn.model_selection import train_test_split

### Шаг 1. Определение функций, которые понадобяться для обучения
# преобразование массива в бинарный вид результатов
def to_one_hot(Y):
    n_col = np.amax(Y) + 1
    binarized = np.zeros((len(Y), n_col))
    for i in range(len(Y)):
        binarized[i, Y[i]] = 1.
    return binarized


# преобразование массива в необходимый вид
def from_one_hot(Y):
    arr = np.zeros((len(Y), 1))

    for i in range(len(Y)):
        l = layer2[i]
        for j in range(len(l)):
            if (l[j] == 1):
                arr[i] = j + 1
    return arr


# сигмоида и ее производная
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x):
    return sigmoid(x) * (1 - sigmoid(x))


# нормализация массива
def normalize(X, axis=-1, order=2):
    l2 = np.atleast_1d(np.linalg.norm(X, order, axis))
    l2[l2 == 0] = 1
    return X / np.expand_dims(l2, axis)


### Шаг 2. Подготовка тренировочных данных
# получения данных из csv файла. укажите здесь путь к файлу Iris.csv
iris_data = pd.read_csv("/Users/ilyalevshenkov/_GB/PYTHON_PLUS/GB_Python/II_Neuro_hw/Iris.csv")

iris_data = pd.read_csv("./Iris.csv")
print(iris_data.head())  # расскоментируйте, чтобы посмотреть структуру данных

# репрезентация данных в виде графиков
g = sns.pairplot(iris_data.drop("Id", axis=1), hue="Species")
plt.show() # расскоментируйте, чтобы посмотреть

# замена текстовых значений на цифровые
iris_data['Species'].replace(['Iris-setosa', 'Iris-virginica', 'Iris-versicolor'], [0, 1, 2], inplace=True)

# формирование входных данных
columns = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
x = pd.DataFrame(iris_data, columns=columns)
x = normalize(x.values)

# формирование выходных данных(результатов)
columns = ['Species']
y = pd.DataFrame(iris_data, columns=columns)
y = y.values
y = y.flatten()
y = to_one_hot(y)

# Разделение данных на тренировочные и тестовые
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.33)

### Шаг 3. Обученние нейронной сети
# присваевание случайных весов
w0 = 2 * np.random.random((4, 5)) - 1  # для входного слоя   - 4 входа, 3 выхода
w1 = 2 * np.random.random((5, 3)) - 1  # для внутреннего слоя - 5 входов, 3 выхода

# скорость обучения (learning rate)
n = 0.001

# массив для ошибок, чтобы потом построить график
errors = []

# процесс обучения
for i in range(5000000):  # 100,000

    # прямое распространение(feed forward)
    layer0 = X_train
    layer1 = sigmoid(np.dot(layer0, w0))
    layer2 = sigmoid(np.dot(layer1, w1))

    # обратное распространение(back propagation) с использованием градиентного спуска
    layer2_error = y_train - layer2
    layer2_delta = layer2_error * sigmoid_deriv(layer2)

    layer1_error = layer2_delta.dot(w1.T)
    layer1_delta = layer1_error * sigmoid_deriv(layer1)

    w1 += layer1.T.dot(layer2_delta) * n
    w0 += layer0.T.dot(layer1_delta) * n

    error = np.mean(np.abs(layer2_error))
    errors.append(error)
    accuracy = (1 - error) * 100

### Шаг 4. Демонстрация полученных результатов
# черчение диаграммы точности в зависимости от обучения
plt.plot(errors)
plt.xlabel('Обучение')
plt.ylabel('Ошибка')
plt.show()  # расскоментируйте, чтобы посмотреть

print("Точность нейронной сети " + str(round(accuracy, 2)) + "%")