import torch as torch
import torch.distributions as dist
import numpy as np

def get_mcmc_proposal(mu, sigma):
    """
    INPUT:
    mu    : scalar
    sigma : tensor, vector of length 7. Should have sigma > 0

    OUTPUT:
    q_mu    : instance of Distribution class, that defines a proposal for mu
    q_sigma : instance of Distribution class, that defines a proposal for sigma
    """

    # YOUR CODE HERE
    # we will use the same distributions as the priors.

    # how do we find the prior stdv on the mean???
    # CHANGE STDEV OF DIST FIGURE IT OUT
    q_mu = dist.Normal(mu, 1)

    q_sigma = dist.Normal(sigma, 0.00001)

    return q_mu, q_sigma


def log_joint(mu, sigma, alpha=50, beta=0.5):
    """
    INPUT:
    mu    : scalar
    sigma : tensor, vector of length 7. Should have sigma > 0
    alpha : scalar, standard deviation of Gaussian prior on mu. Default to 50
    beta  : scalar, rate of exponential prior on sigma_i. Default to 0.5

    OUTPUT:
    log_joint: the log probability log p(mu, sigma, x | alpha, beta), scalar

    NOTE: For inputs where sigma <= 0, please return negative infinity!

    """
    measurements = torch.FloatTensor([-27.020, 3.570, 8.191, 9.898, 9.603, 9.945, 10.056])
    assert mu.ndim == 0
    assert sigma.shape == (7,)

    # YOUR CODE HERE

    # we need to find log(p(mu join data join sigma given a and b))
    # we assume mu and sigma are independent
    for elements in sigma:
        if elements <= 0:
            return torch.tensor(-float("inf"))

    p_sigma = dist.Exponential(rate=beta).log_prob(sigma)
    p_mu = dist.Normal(0, alpha).log_prob(mu)
    p_data = dist.Normal(mu, sigma).log_prob(measurements)

    return torch.sum(p_sigma) + p_mu + torch.sum(p_data)

def mcmc_step(mu, sigma, alpha=50, beta=0.5):
    """
    mu    : scalar
    sigma : tensor, vector of length 7. Should have sigma > 0
    alpha : scalar, standard deviation of Gaussian prior on mu. Default to 50
    beta  : scalar, rate of exponential prior on sigma_i. Default to 0.5

    OUTPUT:
    mu       : the next value of mu in the MCMC chain
    sigma    : the next value of sigma in the MCMC chain
    accepted : a boolean value, indicating whether the proposal was accepted

    """

    accepted = False
    q_mu, q_sigma = get_mcmc_proposal(mu, sigma)

    # YOUR CODE HERE
    # We sample from both distributions

    mu_prop = q_mu.sample()
    sigma_prop = q_sigma.sample()

    q_mu_prop, q_sigma_prop = get_mcmc_proposal(mu_prop, sigma_prop)
    logjoint_prop = log_joint(mu_prop, sigma_prop)
    logjoint = log_joint(mu, sigma)

    logsum = (
            logjoint_prop
            +q_mu_prop.log_prob(mu)
            +torch.sum(q_sigma_prop.log_prob(sigma))
            -logjoint
            -torch.sum(q_sigma.log_prob(sigma_prop))
            -q_mu.log_prob(mu_prop)
    )
    print(type(logsum,),logsum)

    A = min(1, torch.exp_(logsum))
    print(A)
    if A.numpy() > np.random.rand():
        accepted = True
        mu = mu_prop
        sigma = sigma_prop

    return mu, sigma, accepted

measurements = torch.FloatTensor([-27.020, 3.570, 8.191, 9.898, 9.603, 9.945, 10.056])

mu_init = measurements.mean()
sigma_init = torch.ones(7)
mcmc_step(mu_init, sigma_init)
