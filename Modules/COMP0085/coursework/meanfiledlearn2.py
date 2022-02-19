import numpy as np
import matplotlib.pyplot as plt

def compute_ESS(lambd):

    N, K = lambd.shape
    ES = lambd

    ESS = lambd.T @ lambd
    ESS = ESS - np.diag(np.diag(ESS)) + np.diag(np.sum(lambd, axis=0))

    return ES, ESS

def calculate_F(X, mu, sigma, pie, lambd, alpha, b):

    N, D = X.shape
    N_,K = lambd.shape

    # cut-off for lambdas for stability in log operations
    epsilon2 = 1e-12
    lambd[lambd >= 1] = 1 - epsilon2
    lambd[lambd <= 0] = epsilon2

    diagmu = mu.T@mu + np.diag(b)
    ES, ESS = compute_ESS(lambd)

    F = (np.sum(lambd * np.log(pie/lambd) + (1-lambd) * np.log((1-pie)/(1-lambd)))
         -N*D/2 * np.log(2 * np.pi * sigma**2)
         -0.5/sigma**2 * (np.trace(X @ X.T)
                          + np.trace(diagmu @ ESS)
                          - 2 * np.trace(ES.T @ X @ mu)
                          )
         - 0.5 *D*N*np.sum( np.log(2 * np.pi * alpha**(-1)))
         - N* np.sum( alpha/2 * np.diag(diagmu))
         + 0.5 * D * N * np.sum(np.log(2 * np.pi * b))
         + 0.5 * N * K*D
         )

    return F

def MeanField(X, mu, sigma, pie, lambda0, alpha, b, maxsteps):
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
    N_, K = lambda0.shape
    assert N_ == N, "wrong shapes"

    # initialisation
    F_list = []
    lambd = lambda0

    # Stopping criterion
    epsilon = 1e-15

    for iter in range(maxsteps):

        # Iteratively update lambda: we are updating the mean value at all data points for a certain latent dimension
        lambda_new = lambd

        for k in range(K):
            # constants
            diag_mu = np.diag(mu.T @ mu + np.diag(b))

            x = (np.log(pie[:, k]/(1-pie[:, k]))
                 + 1 / (sigma**2)
                 * ((X-lambda_new@mu.T)@mu[:, k]+lambda_new[:, k]*diag_mu[k]
                 - 0.5*diag_mu[k])
                 )
            # we clamp for stability in exponential
            x[x < -700] = -700
            lambda_new[:, k] = 1/(1+np.exp(-x))

            # We also want to update b_i
            b[k] = (np.sum(lambda_new[:, k] / (sigma ** 2) + alpha[k])) ** (-1)

        for k in range(K):
            mu[:, k] = (np.sum(X.T*lambda_new[:, k], axis=1) - np.sum((lambda_new@mu.T).T*lambda_new[:,k], axis=1)
                               + (mu[:, k] * np.sum(lambda_new[:, k]**2))
                        /(np.sum(lambda_new[:, k]) + N * sigma**2 * alpha[k])
            )

        mu[mu > 1e100] = 1e100
        mu[mu < 1e-100] = 1e-100

        # we calculate f with this new lambda, mu, b
        F_new = calculate_F(X, mu, sigma, pie, lambda_new, alpha, b)
        F_list.append(F_new)

        # stopping criterion
        if iter > 100:
            if F_list[iter] - F_list[iter-1] < epsilon:
                break

        lambd = lambda_new

    #plt.plot(F_list)

    return lambd, F_list, mu, b