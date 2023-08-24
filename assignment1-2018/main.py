import autodiff as ad

import numpy as np


x1 = ad.Variable(name="x1")
x2 = ad.Variable(name="x2")

y = x1 * x2 + x1

# print(x1, type(x1))

# print(y, type(y))


grad_x1, grad_x2 = ad.gradients(y, [x1, x2])

print('grad_x1:', grad_x1)
print('grad_x2:', grad_x2)
print('---')

x1_val = 2 * np.ones(1)
x2_val = 3 * np.ones(1)

executor = ad.Executor([y, grad_x1, grad_x2])
y_val, grad_x1_val, grad_x2_val = executor.run(feed_dict={x1: x1_val, x2: x2_val})

print(y_val)
print(grad_x1_val)
print(grad_x2_val)
print('=====')
