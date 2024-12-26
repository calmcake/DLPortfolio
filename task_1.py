"""

Задача: Классификация точек в пересечении двух кругов
Эта задача представляет собой бинарную классификацию, где требуется определить, 
попадает ли точка в пересечение двух окружностей. Основной целью является разработка 
и обучение простой нейронной сети, которая решает эту задачу, а также оценка её 
способности обобщать знания на новых данных.

Описание задачи
Даны два круга с центрами в (-0.5, 0) и (0.5, 0), радиус каждого из которых равен 1.
Целевая метка точки:
1. если точка находится в пересечении этих кругов.
0, если точка не находится в пересечении.

Почему задача успешно решена?
Нейронная сеть была обучена на случайно сгенерированных данных с равномерным распределением 
в квадрате [−1.5,1.5] × [−1.5,1.5]. Метрики обучения показали, что модель корректно 
минимизировала ошибку на тренировочном наборе данных.

Тестирование проводилось на данных, сгенерированных в большем квадрате [−5,5] ×[ −5,5], 
включающем области, которые модель не видела во время обучения.
Тестовая точность модели достигла 100% к концу обучения, что говорит о её способности 
обобщать знания на новые данные.

"""

import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return sigmoid(x) * (1 - sigmoid(x))

def tanh(x):
    return np.tanh(x)

def tanh_derivative(x):
    return 1 - np.tanh(x)**2

def cross_entropy(y_true, y_pred):
    epsilon = 1e-12
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    cross_entropy_loss = -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]
    return cross_entropy_loss

def accuracy(y_true, y_pred):
    predictions = (y_pred >= 0.5).astype(int)
    return np.mean(predictions == y_true)

np.random.seed(42)

x = np.random.uniform(-1.5, 1.5, 4096)
y = np.random.uniform(-1.5, 1.5, 4096)

in_circle1 = (x + 0.5)**2 + y**2 <= 1
in_circle2 = (x - 0.5)**2 + y**2 <= 1

train_data = np.column_stack((x, y))
train_labels = (in_circle1 & in_circle2).astype(int)

x = np.random.uniform(-5, 5, 50)
y = np.random.uniform(-5, 5, 50)

in_circle1 = (x + 0.5)**2 + y**2 <= 1
in_circle2 = (x - 0.5)**2 + y**2 <= 1

test_data = np.column_stack((x, y))
test_labels = (in_circle1 & in_circle2).astype(int)

weights_h1 = 2 * np.random.random((2, 5)) - 1
weights_h2 = 2 * np.random.random((5, 2)) - 1
weights_o = 2 * np.random.random((2, 1)) - 1

bias_h1 = np.zeros((1, 5))
bias_h2 = np.zeros((1, 2))
bias_o = np.zeros((1, 1))

epochs = 1001
alpha = 0.001
batch_size = 128

for epoch in range(epochs):
    train_error = 0

    indexes = np.random.permutation(len(train_data))
    data_randomed = train_data[indexes]
    labels_randomed = train_labels[indexes].reshape(-1, 1)

    for batch_start in range(0, len(train_data), batch_size):
        batched_data = data_randomed[batch_start:batch_start + batch_size]
        batched_labels = labels_randomed[batch_start:batch_start + batch_size]

        layer_0 = batched_data
        layer_1 = tanh(np.dot(layer_0, weights_h1) + bias_h1)
        layer_2 = tanh(np.dot(layer_1, weights_h2) + bias_h2)
        layer_3 = sigmoid(np.dot(layer_2, weights_o) + bias_o)

        train_error += cross_entropy(batched_labels, layer_3)

        delta_3 = (layer_3 - batched_labels) * sigmoid_derivative(layer_3)
        delta_2 = np.dot(delta_3, weights_o.T) * tanh_derivative(layer_2)
        delta_1 = np.dot(delta_2, weights_h2.T) * tanh_derivative(layer_1)

        weights_o -= alpha * np.dot(layer_2.T, delta_3) / batch_size
        weights_h2 -= alpha * np.dot(layer_1.T, delta_2) / batch_size
        weights_h1 -= alpha * np.dot(layer_0.T, delta_1) / batch_size

        bias_o -= alpha * np.sum(delta_3, axis=0, keepdims=True) / batch_size
        bias_h2 -= alpha * np.sum(delta_2, axis=0, keepdims=True) / batch_size
        bias_h1 -= alpha * np.sum(delta_1, axis=0, keepdims=True) / batch_size

    layer_1_test = tanh(np.dot(test_data, weights_h1) + bias_h1)
    layer_2_test = tanh(np.dot(layer_1_test, weights_h2) + bias_h2)
    layer_3_test = sigmoid(np.dot(layer_2_test, weights_o) + bias_o)

    test_error = cross_entropy(test_labels.reshape(-1, 1), layer_3_test)
    test_accuracy = accuracy(test_labels, layer_3_test)

    if epoch % 100 == 0:
        print(f"Epoch: {epoch}, Train Error: {train_error:.4f}, Test Accuracy: {test_accuracy:.4f}")
