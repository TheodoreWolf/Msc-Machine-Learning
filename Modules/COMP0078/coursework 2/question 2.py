import numpy as np
import matplotlib.pyplot as plt

dtrain123 = np.loadtxt("data/dtrain123.dat")
twomoons = np.loadtxt("data/twomoons.dat")

'''
Plotting the Data
'''
twomoons_neg = twomoons[twomoons[:, 0] == -1]
twomoons_pos = twomoons[twomoons[:, 0] == 1]
plt.plot(twomoons[:, 1], twomoons[:, 2], "o")
#plt.plot(twomoons_neg[:, 1], twomoons_neg[:, 2], ".", label="-1")
#plt.plot(twomoons_pos[:, 1], twomoons_pos[:, 2], ".", label="+1")
plt.legend()
plt.show()


def Laplace_matrix(X, c):



    D = np.eye(W.shape[0])
    for i in range(W.shape[0]):
        D[i, i] = np.sum(W[i, :])

    L = D - W

    v, l = np.eig(L)

    second_v = v[:, 1]
    return L, second_v




