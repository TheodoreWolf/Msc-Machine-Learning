import numpy as np
import matplotlib.pyplot as plt

def compute_ESS(lambd):

   # function to compute the ESS give lambda
    N, K = lambd.shape
    ES = lambd

    # ESS is the lambdas squared summed over N with the diagonal switched
    ESS = lambd.T @ lambd
    ESS = ESS - np.diag(np.diag(ESS)) + np.diag(np.sum(lambd, axis=0))

    return ES, ESS

def calculate_F(X, mu, sigma, pie, lambd):

    # function to calculate F
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

def MeanField(X, mu, sigma, pie, lambda0, maxsteps):
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
    epsilon = 1e-100

    # constants
    diag_mu = np.diag(mu.T@mu).flatten()

    for iter in range(maxsteps):

        # Iteratively update lambda: we are updating the mean value at all data points for a certain latent dimension
        lambda_new = lambd
        for k in range(K):
            x = (np.log(pie[:, k]/(1-pie[:, k]))
                 + 1 / (sigma**2)
                 * ((X-lambda_new@mu.T)@mu[:, k]+lambda_new[:, k]*diag_mu[k]
                 - 0.5*diag_mu[k])
                 )
            lambda_new[:, k] = 1/(1+np.exp(-x))

        # we calculate f with this new lambda
        F_new = calculate_F(X, mu, sigma, pie, lambda_new)
        F_list.append(F_new)

        # stopping criterion
        if iter > 5:
            if F_list[iter] - F_list[iter-1] <= epsilon:
                break

        lambd = lambda_new

    #plt.plot(F_list)

    return lambd, F_list