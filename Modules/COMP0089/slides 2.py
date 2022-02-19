import numpy as np
import matplotlib.pyplot as plt
import scipy

samples = np.random.beta(1,2,100000000)
sample2 = np.random.beta(2,1,100000000)
C = np.zeros_like(samples)
C[samples>sample2] = 1
print(C.mean())