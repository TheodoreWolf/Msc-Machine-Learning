import numpy as np
import math
import matplotlib.pyplot as plt
from tqdm import tqdm

def EM(K, X, iterations):
    """
    We define the EM algorithm which takes the number of mixtures k, the dataset X and the number of iterations
    """

    def compute_likelihood(N, D, X, probabilities, piks):
        # Likelihood function, logged
        likelihood = 0
        for n in range(N):
            sum_bern = 0
            for k in range(K):
                bernoulli = 1
                for d in range(D):
                    bernoulli *= probabilities[d, k]**(X[n, d]) * (1-probabilities[d, k])**(1 - X[n, d])
                sum_bern += bernoulli * piks[k, :]
            likelihood += np.log(sum_bern)
        return likelihood

    def E_step(N, D, X, piks, probabilities):

        # Computing responsibilities: E step
        responsibilities = np.zeros((N, K))

        for n in range(N):
            # The denominator for each n is the same, we only need to compute it once per n
            denominator = 0
            for k in range(K):
                # resetting the Bernoulli variable
                bernoulli = 1
                for d in range(D):
                    # computing the bernoulli for a given n, k and d
                    bernoulli *= probabilities[d, k]**(X[n, d]) * (1 - probabilities[d, k])**(1 - X[n, d])

                # We then increase the denominator with the bernoulli over all d
                denominator += bernoulli * piks[k, :]

                # And compute the un-normalised responsibilities for given [n,k]
                responsibilities[n, k] = piks[k, :] * bernoulli

            # For a given n we normalise the responsibilities across all ks
            responsibilities[n, :] = responsibilities[n, :] / denominator

        return responsibilities

    def M_step(N, D, X, piks, probabilities, responsibilities):
        # Computing the new piks and new probabilities: M step

        # New probabilities
        for k in range(K):
            sum_resp = np.sum(responsibilities[:, k])
            for d in range(D):
                probabilities[d, k] = np.sum(np.dot(responsibilities[:, k], X[:, d])) / sum_resp

        # New piks
        for k in range(K):
            piks[k, :] = np.sum(responsibilities[:, k]) / N

        return probabilities, piks

    # Throughout this code it is good to think as: K the number of mixtures chosen by the user,
    # N the number of images in the dataset and D the number of dimensions for each image (pixels).
    # We will index these with their lowercase equivalents.
    N, D = X.shape

    # initialising with random values
    # np.random.seed(1)
    probabilities = np.random.rand(D, K)

    #  We define a cut off so that we end the iterations if the log-likelihood converges
    cut_off = 0

    # The entries of the piks (pi index k plural) need to sum to one: we can use a random Dirichlet sample
    piks = np.random.dirichlet(np.ones([K]), size=1).T

    # Storing the log-likelihoods in a list for easy plotting and computing the first one
    likelihoods = [compute_likelihood(N, D, X, probabilities, piks)]

    for i in tqdm(range(iterations)):

        # E-step first
        responsibilities = E_step(N, D, X, piks, probabilities)

        # Then M-step
        probabilities, piks = M_step(N, D, X, piks, probabilities, responsibilities)

        # Compute the log-likelihood
        likelihoods.append(compute_likelihood(N, D, X, probabilities, piks))

        # check for convergence after the first loop
        if i > 10:
            if likelihoods[i]-likelihoods[i-1] < cut_off:
                print("Reached cut-off after {} iterations".format(i))
                break

    return likelihoods, probabilities, responsibilities, piks


if __name__ == "__main__":
    # We load the data
    X = np.loadtxt("http://www.gatsby.ucl.ac.uk/teaching/courses/ml1/binarydigits.txt")

    # We choose the number of mixtures K and iterations I
    K = 3
    I = 1000

    # We run the EM algorithm, outputs are the log-likelihood, the probability matrix, the responsibilities and
    # the mixture probability respectively
    l, p, r, pi = EM(K, X, I)

    # We can plot the different probabilities for each mixture in the same way as in binarydigits.py
    for i in range(K):
        plt.subplot(1, K, i + 1)
        plt.imshow(np.reshape(p[:, i], (8, 8)),
                   interpolation="None",
                   cmap='gray')
        plt.axis('off')
    plt.show()


