#2. Подайте на вход сети 10 новых примеров, которые не были задействованы в обучении, оцените численно ее точность предсказания
# (то есть сравниваем результаты predict и верную отметку на тесте).

import numpy as np

# Пример реализации класса нейронной сети
class NeuralNetwork:
    def __init__(self):
        # Инициализация весов и параметров сети
        self.weights = np.random.rand(2, 1)
        self.bias = np.random.rand(1)

    def predict(self, input_data):
        # Прямой проход через сеть
        output = np.dot(input_data, self.weights) + self.bias
        # Округление выхода до 0 или 1
        rounded_output = np.round(output)
        return rounded_output.astype(int)  # Приводим к типу int для удобства

    def evaluate_accuracy(self, test_data, test_labels):
        # Получение предсказаний для тестовых данных
        predictions = self.predict(test_data)
        # Подсчет числа правильных предсказаний
        correct_predictions = np.sum(predictions == test_labels)
        # Вычисление точности
        accuracy = correct_predictions / len(test_labels)
        return accuracy

# Пример тестовых данных и их меток
test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [0, 1], [1, 0], [1, 1], [0, 0], [0, 0], [1, 1]])
test_labels = np.array([0, 1, 1, 0, 1, 1, 0, 0, 0, 1])  # Предположим, что это верные метки для тестовых данных

# Создание экземпляра нейронной сети
model = NeuralNetwork()

# Оценка точности на новых данных
accuracy = model.evaluate_accuracy(test_data, test_labels)
print(f"Accuracy on test data: {accuracy}")
