#1. Сделаете метод класса сети predict, который округляет выход последнего нейрона.

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

# Пример использования нейронной сети для классификации
def main():
    # Создание экземпляра нейронной сети
    model = NeuralNetwork()
    # Входные данные для классификации
    input_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    # Получение предсказаний для входных данных
    predictions = model.predict(input_data)
    # Вывод результатов
    for i in range(len(predictions)):
        print(f"Input: {input_data[i]}, Prediction: {predictions[i]}")

if __name__ == "__main__":
    main()
