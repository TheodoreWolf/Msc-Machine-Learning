import numpy as np
import scipy.special as sp

X = np.loadtxt("http://www.gatsby.ucl.ac.uk/teaching/courses/ml1/binarydigits.txt")
"""
Model 1
"""
print("Model 1 log likelihood: ", np.log(0.5)*6400-np.log(3))
"""
Model 2 
"""
sumx = np.sum(X)
print("Model 2 log likelihood: ", sp.betaln(1+sumx, 1+6400-sumx)-np.log(3))

"""
Model 3
"""
loglike = 0
for d in range(64):
    sumn = np.sum(X[:, d])
    loglike += sp.betaln(1+sumn, 1+100-sumn)
print("Model 3 log likelihood: ", loglike-np.log(3))
