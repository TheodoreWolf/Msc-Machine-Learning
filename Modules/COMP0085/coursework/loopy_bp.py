import numpy as np
import matplotlib.pyplot as plt

def compute_ESS(lambd):

    N, K = lambd.shape
    ES = lambd

    ESS = lambd.T @ lambd
    ESS = ESS - np.diag(np.diag(ESS)) + np.diag(np.sum(lambd, axis=0))

    return ES, ESS

def calculate_F(X, mu, sigma, pie, lambd):

    N, D = X.shape
    N_,K = lambd.shape

    # cut-off for lambdas for stability in log operations
    epsilon2 = 1e-12
    lambd[lambd >= 1] = 1 - epsilon2
    lambd[lambd <= 0] = epsilon2

    ES, ESS = compute_ESS(lambd)

    F = (np.sum(lambd * np.log(pie/lambd) + (1-lambd) * np.log((1-pie)/(1-lambd)))
         -N*D/2 * np.log(2 * np.pi * sigma**2)
         -0.5/sigma**2 * (np.trace(X @ X.T)
                          + np.trace(mu.T @ mu @ ESS)
                          - 2 * np.trace(ES.T @ X @ mu)
                          )
         )

    return F

def EP(X, mu, sigma, pie, message0, maxsteps):
    '''

    :param X: (NxD) data matrix
    :param mu: (DxK) matrix of means
    :param sigma: standard deviation of data
    :param pie: (1xK) vector of priors on s
    :param lambda0: (NxK) initial values for lambda
    :param maxsteps: Maximum number of steps for the fixed point equations
    :return: lambd: (NxK)
    :return: F: scalar, lower bound on Likelihood for each datapoint
    '''
    # Dimensions
    N, D = X.shape
    K, K, N_ = message0.shape
    assert N_ == N, "wrong shapes"
    assert mu.shape[1] == K

    # initialisation
    F_list = []
    message = np.zeros((K,K,N))

    # Stopping criterion
    epsilon = 1e-100

    # constants
    diag_mu = np.diag(mu.T@mu).flatten()

    # we set the natural parameters of f_i
    f = np.zeros((N,K))

    for n in range(N):
        f[n, :] = np.log(pie / (1 - pie)) + X[n, :]@mu/(sigma**2) - diag_mu/(2*sigma**2)

    for iter in range(maxsteps):
        #message = message0
        for i in range(K):
            for j in range(i + 1, K):

                # we use a damping parameter
                a = 0

                # update the message from j to i
                omega0 = f[:, j] + np.sum(message[:, j, :], axis=0) - message[i, j, :]
                W = -mu[:, i].T @ mu[:, j] / (sigma ** 2)

                # we use our equation for the update to omega
                omega1 = (np.exp(omega0 + W) + 1) / (1 + np.exp(omega0))

                # we use our damping parameter to help convergence
                message[j, i, :] = a * message[j, i, :] + (1 - a) * np.log(omega1)

                # update the message from i to j (reverse message)
                omega0 = f[:, i] + np.sum(message[:, i, :], axis=0) - message[j, i, :]
                W = -mu[:, j].T @ mu[:, i] / (sigma ** 2)

                # we use our equation for the update to omega
                omega1 = (np.exp(omega0 + W) + 1) / (1 + np.exp(omega0))

                # we use our damping parameter to help convergence
                message[i, j,:] = a * message[i, j,:] + (1 - a) * np.log(omega1)

        lambdas = np.zeros((N, K))
        for n in range(N):
            eta = f[n, :]+np.sum(message[:,:,n], axis=0)
            lambdas[n, :] = 1/(np.exp(-eta) + 1)

        # we calculate f with this new lambda
        F_new = calculate_F(X, mu, sigma, pie, lambdas)
        F_list.append(F_new)

    return lambdas, F_list, message