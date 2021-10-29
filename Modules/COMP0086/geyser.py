import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm

geyser_data = np.loadtxt(os.path.join(r"C:\Users\theod\Desktop\UCL\Machine learning\COMP0086", 'geyser.txt'))
plt.figure()
#plt.plot(geyser_data[:, 0], geyser_data[:, 1], "o")
for n in tqdm(range(0, 294)):
    plt.plot(geyser_data[1, -1-n:1], geyser_data[n+1, -1:1], "o")
plt.show()