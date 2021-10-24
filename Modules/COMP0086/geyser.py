import numpy as np
import os
import matplotlib.pyplot as plt

geyser_data = np.loadtxt(os.path.join(r"C:\Users\theod\Desktop\UCL\Machine learning\COMP0086", 'geyser.txt'))
plt.figure()
plt.plot(geyser_data[:, 0], geyser_data[:, 1], "o")
plt.show()
