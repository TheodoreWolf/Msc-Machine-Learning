import numpy as np
import matplotlib.pyplot as plt

# we import all our functions
from m_step import M_step
from Mean_field_learning import MeanField, compute_ESS, calculate_F
from genimages import generate_data


def LearnBinFactors(X, K, iterations):

    # dimensions
    N, D = X.shape

    # initialisation
    epsilon = 1e-100
    F_list = []
    maxsteps = 300

    # 1, 10, seed 101 with 800 datapoints for almost perfect, 6700 with 400 for 7/8 ,
    # 6009 with 400 for rly good, 0:400, 789 for almost perfect
    np.random.seed(789)
    lambda0 = np.random.rand(N, K)
    ES, ESS = compute_ESS(lambda0)
    mu, sigma, pie = M_step(X, ES, ESS)
    lambd = lambda0


    for i in range(iterations):

        # E-step first
        lambd, F = MeanField(X, mu, sigma, pie, lambd, maxsteps)

        # we update our expectations
        ES, ESS = compute_ESS(lambd)

        # M-step next
        mu, sigma, pie = M_step(X, ES, ESS)

        # we calculate the free energy
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

    # plotting the free energy
    plt.figure()
    plt.plot(F_list)
    plt.xlabel("Iteration")
    plt.ylabel("Variational Free Energy")

    print(F_list[-1])
    return mu, sigma, pie, lambd

if __name__ == "__main__":

    X, original = generate_data(400, 16)
    K = 8
    mu, sigma, pie, lambd = LearnBinFactors(X, K, 300)

    # we show the learned mean parameters
    plt.figure()
    for k in range(K):
        plt.subplot(int(K/4), int(K/2), k + 1)
        plt.imshow(np.reshape(mu[:, k], (4, 4)), cmap=plt.gray(), interpolation='none')
        plt.axis('off')
    plt.suptitle("Learned mean parameter")

    # the original patterns for comparison
    # plt.figure()
    # for k in range(8):
    #     plt.subplot(2, 4, k + 1)
    #     plt.imshow(np.reshape(original[k], (4, 4)), cmap=plt.gray(), interpolation='none')
    #     plt.title("Original patterns")
    #     plt.axis('off')

    # we calculate the free energy with learned mean parameters except sigma
    plt.figure()
    sigma = [1,5, 10]
    for s in sigma:
        l, f = MeanField(X[0,:][None, :], mu, s, pie, lambd[0,:][None,:], 100)

        # free energy
        plt.plot(f, label="$\sigma$={}".format(s))
        plt.legend()
        plt.ylabel("$F(t)$")
        plt.xlabel("Mean field iteration")
        plt.grid()

        # log(f(t)-f(t-1)
        # loglist = []
        # for t in range(len(f)):
        #     if t > 0:
        #         loglist.append(np.log(f[t] - f[t - 1]+1e-10))
        # plt.plot(loglist, label="$\sigma$={}".format(s))
        # plt.legend()
        # plt.ylabel("$log(F(t)-F(t-1) + 10^{-10})$")
        # plt.xlabel("Mean field iteration")
        # plt.grid()

    plt.show()