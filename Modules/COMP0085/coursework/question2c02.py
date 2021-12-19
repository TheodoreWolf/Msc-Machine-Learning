import numpy as np
import matplotlib.pyplot as plt

data = np.loadtxt("http://www.gatsby.ucl.ac.uk/teaching/courses/ml1-2012/co2.txt")
labels = data[:, 2]
# Asssuming the first of each month
time = data[:, 0] + (data[:,1]-1) /12

plt.plot(time, labels)
plt.show()