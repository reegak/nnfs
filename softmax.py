import data
import numpy as np
import feedforward
import relu
class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims= True)
        self.output = probabilities


X, y = data.create_data(points = 100, classes = 3)

dense1 = feedforward.Layer_Dense(2,3)
activation1 = relu.Activation_ReLU()

dense2 =feedforward.Layer_Dense(3,3)
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)

dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])
