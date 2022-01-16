import numpy as np

x = np.array([
[9, 2, 5, 0, 0],
[7, 5, 0, 0 ,0]])
x_exp = np.exp(x)

x_sum = np.sum(x_exp, axis=1, keepdims=True)
print(str(x_sum) + str(x_sum.shape))
# s = x_exp / x_sum