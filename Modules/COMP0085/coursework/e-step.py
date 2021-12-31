import numpy as np

def e_step(ES):
    lambd = ES

    return lambd

def calculate_F(X, mu, sigma, pie, lambd, ES, ESS):

    N, D = X.shape

    F = (np.sum(ES * np.log(pie) + (1-ES) * np.log(1-pie), axis=1)
         -D/2 * np.log(2 * np.pi * sigma**2)
         -0.5/sigma**2 * (np.trace(X.T @ X) + np.trace(mu.T @ mu @ ESS)) - 2 * np.trcae(mu.T @ ES @ X)
         -np.sum(ES * np.log(lambd) + (1-ES) * np.log(1-lambd), axis=1)



         )
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

    # initialisation
    F = []
    lambd = lambda0

    epsilon = 0
    # we want to do e-step then m-step

    # we want ESS and ES


    for iter in range(maxsteps):

        F_iter = calculate_F(X, mu, sigma, pie, lambd)
        F.append(F_iter)
        if F[iter]-F[iter-1] <= epsilon:
            break


    return lambd, F