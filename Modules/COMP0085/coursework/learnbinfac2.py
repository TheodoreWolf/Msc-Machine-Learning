import numpy as np
import matplotlib.pyplot as plt

from m_step import M_step, M_step_alpha
from meanfiledlearn2 import MeanField, compute_ESS, calculate_F
from genimages import generate_data


def LearnBinFactors(X, K, iterations, params=None):

    # dimensions
    N, D = X.shape

    # initialisation
    epsilon = 1e-100
    F_list = []
    maxsteps = 300

    np.random.seed(10)
    lambda0 = np.random.rand(N, K)
    mu = np.random.rand(D,K)
    b = np.ones((K,))
    alpha = np.ones((K,))*1000000
    pie = np.random.rand(1,K)
    sigma = np.random.rand()
    # ES, ESS = compute_ESS(lambda0)
    # alpha, sigma, pie = M_step_alpha(X, ES, ESS, mu, b)

    lambd = lambda0

    for i in range(iterations):

        # E-step first
        lambd, F, mu, b = MeanField(X, mu, sigma, pie, lambd, alpha, b, maxsteps)

        # we update our expectations and add F to our list
        ES, ESS = compute_ESS(lambd)

        # M-step next
        alpha, sigma, pie = M_step_alpha(X, ES, ESS, mu, b)

        F_mstep = calculate_F(X, mu, sigma, pie, lambd, alpha, b)
        F_list.append(F_mstep)

        print(K, "Iteration number {} with free energy {:.4f}".format(i, F_mstep))

        # stopping criterion with minimum number of iterations
        if i > 100:

            if F_list[-1]-F_list[-2] < epsilon:
                print("Reached cut-off after {} iterations".format(i))
                break
            # check for increase in F
            #assert F_list[-1] >= F_list[-2]

    print(F_list[-1])
    print(b)
    return mu, sigma, pie, lambd, alpha, F_list

if __name__ == "__main__":

    X, original = generate_data(800, 16)
    K = 32
    mu, sigma, pie, lambd, alpha, f = LearnBinFactors(X, K, 300)

    plt.figure()
    for k in range(K):
        plt.subplot(4, 10, k + 1)
        plt.imshow(np.reshape(mu[:, k], (4, 4)), cmap=plt.gray(), interpolation='none')
        plt.axis('off')
    plt.suptitle("Learned mean parameter")

    plt.figure()
    x = [i for i in range(K)]
    plt.plot(x, np.sort(alpha),".")
    plt.yscale("log")
    plt.grid()
    plt.ylabel("Sorted $\\alpha_i$ values")
    plt.xlabel("Latent dimension K")

    plt.figure()
    plt.plot(f)
    plt.grid()

    plt.ylabel("Variational free energy")
    plt.xlabel("Iteration")

    # K = [2,4,8,12,16,32]
    # plt.figure()
    # for k in K:
    #     mu, sigma, pie, lambd, alpha, f = LearnBinFactors(X, k, 300)
    #     plt.plot(f[20:], label="K={}".format(k))
    #     # x = [i for i in range(k)]
    #     # plt.plot(x, np.sort(alpha),".", label="K={}".format(k))

    plt.show()