import matplotlib.pyplot as plt
import nn_lib as nn
import numpy as np
X, y = nn.create_data(100,3)

dense1 = nn.Layer_Dense(2,3)
activation1 = nn.Activation_ReLU()
dense2 = nn.Layer_Dense(3,3)
activation2 = nn.Activation_Softmax()
loss_fx = nn.Loss_CategoricalCrossentropy()
lowest_loss = 999999

best_dense1_weights = dense1.weights.copy()
best_dense1_biases = dense1.biases.copy()
best_dense2_weights = dense2.weights.copy()
best_dense2_biases = dense2.biases.copy()

for i in range(1000000):
    dense1.weights += .05*np.random.randn(2,3)
    dense1.biases += .05 * np.random.randn(1,3)
    dense2.weights += .05 * np.random.randn(3,3)
    dense2.biases += .05 * np.random.randn(1,3)

    dense1.forward(X)
    activation1.forward(dense1.output)
    dense2.forward(activation1.output)
    activation2.forward(dense2.output)

    loss = loss_fx.calculate(activation2.output, y)

    predictions = np.argmax(activation2.output, axis = 1)
    accuracy = np.mean(predictions == y)

    if loss < lowest_loss:
        print('New set of weights found, iteration:', i, 'loss:', loss, 'acc:', accuracy)
        best_dense1_weights = dense1.weights.copy()
        best_dense1_biases = dense1. biases.copy()
        best_dense2_weights = dense2.weights.copy()
        best_dense2_biases = dense2.biases.copy()
        lowest_loss = loss

