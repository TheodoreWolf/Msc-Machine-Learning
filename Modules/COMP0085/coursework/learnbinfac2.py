import numpy as np
import matplotlib.pyplot as plt

from m_step import M_step, M_step_alpha, re_update
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
    sigma = 2
    pie = np.random.rand(1, K)
    alpha= np.ones((K,))
    lambd = lambda0

    for i in range(iterations):

        # E-step first
        lambd, F = MeanField(X, mu, sigma, pie, lambd, alpha, maxsteps)

        # we update our expectations and add F to our list
        ES, ESS = compute_ESS(lambd)

        # M-step next
        for j in range(200):
            mu2, sigma2, pie = M_step_alpha(X, ES, ESS, alpha, sigma, mu)

            alpha2 = re_update(X, mu2, ES, ESS)

            if sigma2 == sigma and j >= 10:
                print("Broke m-step after {} iters".format(j))
                sigma = sigma2
                mu = mu2
                alpha = alpha2
                break
            sigma = sigma2
            mu = mu2
            alpha = alpha2

        F_mstep = calculate_F(X, mu, sigma, pie, lambd, alpha)
        F_list.append(F_mstep)

        print(K, "Iteration number {} with free energy {:.4f}".format(i, F_mstep))

        # stopping criterion
        if i > 100:
            if F_list[-1]-F_list[-2] < epsilon:
                print("Reached cut-off after {} iterations".format(i))
                break
            # check for increase in F
            assert F_list[-1] >= F_list[-2]



    print(F_list[-1])

    return mu, sigma, pie, lambd, alpha, F_list

if __name__ == "__main__":

    X, original = generate_data(1600, 16)

    #mu, sigma, pie, lambd, alpha = LearnBinFactors(X, K, 300)

    #print(alpha)
    # plt.figure()
    # for k in range(K):
    #     plt.subplot(int(K/4), int(K/2), k + 1)
    #     plt.imshow(np.reshape(mu[:, k], (4, 4)), cmap=plt.gray(), interpolation='none')
    #     plt.axis('off')
    # plt.suptitle("Learned mean parameter")

    # plt.figure()
    # for k in range(K):
    #     plt.subplot(10, 10, k + 1)
    #     plt.imshow(np.reshape(mu[:, k], (4, 4)), cmap=plt.gray(), interpolation='none')
    #     plt.axis('off')
    # plt.suptitle("Learned mean parameter")

    # plt.figure()
    # x = [i for i in range(K)]
    # plt.plot(x, np.sort(alpha),".")
    # plt.yscale("log")
    # plt.grid()
    # plt.ylabel("Sorted $\log(\\alpha_i)$ values")
    # plt.xlabel("Latent dimension K")

    K = [2,4,8,12,16,32]
    plt.figure()
    for k in K:
        mu, sigma, pie, lambd, alpha, f = LearnBinFactors(X, k, 300)
        # plt.plot(f, label="K={}".format(k))
        x = [i for i in range(k)]
        plt.plot(x, np.sort(alpha),".", label="K={}".format(k))


    plt.legend()
    plt.yscale("log")
    plt.grid()
    plt.ylabel("Sorted $\log(\\alpha_i)$ values")
    plt.xlabel("Latent dimension K")

    plt.show()