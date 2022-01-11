import numpy as np
import matplotlib.pyplot as plt

from m_step import M_step
from loopy_bp import EP, compute_ESS, calculate_F
from genimages import generate_data


def LearnBinFactors(X, K, iterations):

    # dimensions
    N, D = X.shape

    # initialisation
    epsilon = 1e-100
    F_list = []
    maxsteps = 300

    # 1, 10, seed 101 with 800 datapoints for almost perfect, 6700 with 400 for 7/8 , 6009 with 400 for rly good, 0:400, 789
    np.random.seed(789)
    lambda0 = np.random.rand(N, K)
    ES, ESS = compute_ESS(lambda0)
    mu, sigma, pie = M_step(X, ES, ESS)
    lambd = lambda0
    message0 = np.random.rand(K,K,N)
    F_list.append(calculate_F(X, mu, sigma, pie, lambd))

    for i in range(iterations):

        # E-step first
        lambd, F, message = EP(X, mu, sigma, pie, message0, maxsteps)

        # we update our expectations and add F to our list
        ES, ESS = compute_ESS(lambd)

        # M-step next
        mu, sigma, pie = M_step(X, ES, ESS)

        F_mstep = calculate_F(X, mu, sigma, pie, lambd)
        F_list.append(F_mstep)

        print("Iteration number {} with free energy{:.4f}".format(i, F_mstep))

        message0 = message
        if i > 10:
            if abs(np.mean(F_list[10:i])) < epsilon:
                print("Reached cut-off after {} iterations".format(i))
                break
    var = plt.figure()
    plt.plot(F_list)
    plt.xlabel("Iteration")
    plt.ylabel("Variational Free Energy")

    print(F_list[-1])
    return mu, sigma, pie, lambd

if __name__ == "__main__":

    X, original = generate_data(400, 16)
    K = 8
    mu, sigma, pie, lambd= LearnBinFactors(X, K, 100)

    plt.figure()
    for k in range(K):
        plt.subplot(int(K/4), int(K/2), k + 1)
        plt.imshow(np.reshape(mu[:, k], (4, 4)), cmap=plt.gray(), interpolation='none')
        plt.axis('off')
    plt.suptitle("Learned mean parameter")
    plt.show()
    # plt.figure()
    # for k in range(8):
    #     plt.subplot(2, 4, k + 1)
    #     plt.imshow(np.reshape(original[k], (4, 4)), cmap=plt.gray(), interpolation='none')
    #     plt.title("Original patterns")
    #     plt.axis('off')

    # plt.figure()
    # sigma = [1,5, 10]

    plt.show()