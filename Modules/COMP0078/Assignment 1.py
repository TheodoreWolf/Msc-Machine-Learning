from matplotlib import pyplot as plt
import numpy as np

n = 4
xdata = [1,2,3,4]
ydata = [3,2,0,5]
x = np.array([xdata])
X = x**0
for order in range(1,n):
  X = np.append(X, x**order, axis=0)
  print("xorder",x**order)
  print("matrix",X)