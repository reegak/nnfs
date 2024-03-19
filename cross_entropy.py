import numpy as np
import math
softmax_ouput = [0.7,.1,2]
target_output = [1,0,0]

loss = -(math.log(softmax_ouput[0])*target_output[0] +
        math.log(softmax_ouput[1])*target_output[1] +
        math.log(softmax_ouput[2])*target_output[2 ] )
print(loss)
# print(-math.log(.7))
# print(-math.log(.5))
#loss = -math.log(softmax_ouput[0])
# print(loss)