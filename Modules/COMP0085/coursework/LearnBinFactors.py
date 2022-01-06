import numpy as np
import matplotlib.pyplot as plt

from m_step import M_step
from Mean_field_learning import MeanField, compute_ESS, calculate_F
from genimages import generate_data


def LearnBinFactors(X, K, iterations, params=None):

    # dimensions
    N, D = X.shape

    # initialisation
    epsilon = 1e-100
    F_list = []
    maxsteps = 300

    if params is None:
        # 1, 10, seed 101 with 800 datapoints for almost perfect, 6700 with 400 for 7/8 , 6009 with 400 for rly good, 0:400
        np.random.seed(101)
        lambda0 = np.random.rand(N, K)
        ES, ESS = compute_ESS(lambda0)
        mu, sigma, pie = M_step(X, ES, ESS)
        lambd = lambda0
    else:
        lambd, mu, sigma, pie = params

    for i in range(iterations):

        # E-step first
        lambd, F = MeanField(X, mu, sigma, pie, lambd, maxsteps)

        # we update our expectations and add F to our list
        ES, ESS = compute_ESS(lambd)


        # M-step next
        mu, sigma, pie = M_step(X, ES, ESS)


        F_mstep = calculate_F(X, mu, sigma, pie, lambd)
        F_list.append(F_mstep)

        print("Iteration number {} with free energy{:.4f}".format(i, F_mstep))

        # stopping criterion
        if i > 2:
            if F_list[-1]-F_list[-2] < epsilon:
                print("Reached cut-off after {} iterations".format(i))
                break
            # check for increase in F
            assert F_list[-1] >= F_list[-2]

    plt.figure()
    plt.plot(F_list)
    plt.xlabel("Iteration")
    plt.ylabel("Variational Free Energy")

    print(F_list[-1])
    return mu, sigma, pie, lambd

if __name__ == "__main__":

    X, original = generate_data(800, 16)
    K = 8
    mu, sigma, pie, lambd = LearnBinFactors(X, K, 300)

    plt.figure()
    for k in range(K):
        plt.subplot(int(K/4), int(K/2), k + 1)
        plt.imshow(np.reshape(mu[:, k], (4, 4)), cmap=plt.gray(), interpolation='none')
        plt.axis('off')
    plt.suptitle("Learned mean parameter")

    # plt.figure()
    # for k in range(8):
    #     plt.subplot(2, 4, k + 1)
    #     plt.imshow(np.reshape(original[k], (4, 4)), cmap=plt.gray(), interpolation='none')
    #     plt.title("Original patterns")
    #     plt.axis('off')

    plt.figure()
    l, f = MeanField(X[0,:][None, :], mu, sigma, pie, lambd[0,:][None,:], 100)
    plt.figure()

    for t in range(len(f)):
        if t > 0:
            plt.plot(np.log(f[t] - f[t - 1]))

    plt.show()