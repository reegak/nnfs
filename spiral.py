import matplotlib.pyplot as plt
import feedforward as ff
import data
import relu
import cross_entropy as ce
import loss
X,y = data.create_data(100,3)

dense1 = ff.Layer_Dense(2,3)
activation1 = relu.Activation_ReLU()
dense2 = ff.Layer_Dense(3,3)

loss_fx = loss.Loss_CategoricalCrossentropy()
lowest_loss = 9999999
