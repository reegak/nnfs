import numpy as np

softmax_outputs = np.array([[.7,.1,.2],
                            [.2,.5,.4],
                            [.02,.9,.08]])

class_targets = [0,1,1]
print((-np.log(softmax_outputs[[0,1,2],[class_targets]])))


import numpy as np
import data
X = [[1,2,3,2.5],
          [2.0,5.0,-1.0,2.0],
          [-1.5,2.7,3.3,-.8]]

X, y = data.create_data(100,3)

class Activation_Softmax:
    def forward(self,inputs):
        exp_values = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis = 1, keepdims= True)
        self.output = probabilities
class Layer_Dense:
    def __init__(self, n_inputs,n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs,n_neurons)
        self.biases = np.zeros((1,n_neurons))
    def forward(self,inputs):
        self.output = np.dot(inputs,self.weights) + self.biases

class Activation_ReLU:
    def forward(self,inputs):
        self.output = np.maximum(0,inputs)

class Loss:
    def calculate(self, output, y):
        sample_losses = self.forward(output,y)
        data_loss = np.mean(sample_losses)
        return data_loss
    

class Loss_CategoricalCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 1e-7,1-1e-7)

        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped[range(samples), y]
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true, axis =1)

        negative_log_like = -np.log(correct_confidences)
        return negative_log_like

activation2 = Activation_Softmax()
layer1 = Layer_Dense(2,3)
activation1 = Activation_ReLU()
layer1.forward(X)
activation1.forward(layer1.output)
dense2 = Layer_Dense(3,3)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

loss_fx = Loss_CategoricalCrossentropy()
loss = loss_fx.calculate(activation2.output, y)