# PCA
import math
import random
import matplotlib.pyplot as plt
import numpy as np
import time
import pandas as pd
iris = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data",
                  header=None)
iris.columns = ["sepal_length","sepal_width",
                'petal_length','petal_width','species']
iris.dropna(how='all', inplace=True)
iris.head()


def standardise_data(arr):
    rows, columns = arr.shape
    X = arr

    standardisedArray = np.zeros(shape=(rows, columns))
    tempArray = np.zeros(rows)

    for column in range(columns):

        mean = np.mean(X[:, column])
        std = np.std(X[:, column])
        tempArray = np.empty(0)

        for element in X[:, column]:
            tempArray = np.append(tempArray, ((element - mean) / std))

        standardisedArray[:, column] = tempArray

    return standardisedArray


X = iris.iloc[:, 0:4].values
X.shape

# 1 standardise data
X = standardise_data(X) # X : m * n matrix, m samples, n dimensions

# 2 compute variance matrix
cov = np.cov(X.T)

# 3 compute eigenvalues
eigenvals, eigenvecs = np.linalg.eig(cov)
eigenvecs = eigenvecs.T

# 4 project to principal components respectively?
plt.scatter(np.dot(X, eigenvecs[0]), np.dot(X, -eigenvecs[1]), marker="_")
#plt.show() # this is wrong

# reference: scikitlearn
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
pcs = pca.fit_transform(X)
pcsdf = pd.DataFrame(data=pcs, columns=['pc 1', 'pc 2'])
plt.scatter(pcsdf['pc 1'], pcsdf['pc 2'], marker="|")
plt.show()