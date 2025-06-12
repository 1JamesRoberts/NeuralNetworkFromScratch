import numpy as np



a = np.array(np.arange(3)).reshape(3)
w = np.array(np.arange(9)).reshape(3,3)
b = np.array(np.arange(3)).reshape(3)
c = np.array(np.ones((1,3)))
print(a*b)
print(np.dot(a,b))
