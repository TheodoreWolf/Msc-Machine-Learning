import numpy as np
import scipy.special as sp

X = np.loadtxt("http://www.gatsby.ucl.ac.uk/teaching/courses/ml1/binarydigits.txt")

"""
model 2 
"""
sumx = np.sum(X)
print(sp.betaln(1+sumx, 1+6400-sumx)-np.log(3))