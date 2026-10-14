import numpy as np


def softmax(x):
    x_shifted = x - np.max(x)  
    exp_x = np.exp(x_shifted)
    return exp_x / np.sum(exp_x)

a = np.asarray([2.5, -1.3, 0.7, 4.2, -0.5, 1.1, 0.3])
a = a - np.max(a)
a = np.exp(a)
b = a / np.sum(a)
print(b) 
# b = softmax(a)
# print(b)