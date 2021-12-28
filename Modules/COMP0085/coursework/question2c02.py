import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist

# data = np.loadtxt("http://www.gatsby.ucl.ac.uk/teaching/courses/ml1-2012/co2.txt")
# labels = data[:, 2]
# # Asssuming the first of each month
# time = data[:, 0] + (data[:,1]-1) /12

data = np.loadtxt("http://www.gatsby.ucl.ac.uk/teaching/courses/ml1/co2.txt")

trend = data[:, 4]
labels = data[:, 3]
time = data[:, 2]
features = np.concatenate((time[:, None], np.ones((time.shape))[:, None]), axis=1)
prior_cov = np.array([[100, 0], [0, 10000]])
post_cov = np.linalg.pinv(features.T @ features + np.linalg.pinv(prior_cov))
post_mean = post_cov @ features.T @ labels



def question_a():
    np.random.seed(0)
    post = dist.MultivariateNormal(torch.tensor(post_mean), torch.tensor(post_cov))
    normaliser = np.sqrt(np.linalg.det(post_cov)/(2*np.pi*np.linalg.det(prior_cov)))
    exponent = np.exp(-0.5*labels[:, None].T @ (np.eye(features.shape[0])-(features@post_cov@features.T))@ labels[:, None])
    # weights = post.sample((1, 4)).numpy().squeeze()
    # plt.plot(time, (weights@features.T).T)
    print(post_cov)
    print(post_mean)
    '''
    Using least squares
    '''

    a, b = np.linalg.pinv(features.T @ features) @ (features.T @ labels)
    plt.figure()
    plt.plot(time, a * time + b, label="least squares")

    plt.plot(time, labels, label="data")
    plt.plot(time, time * post_mean[0] + post_mean[1], label="posterior mean weight")
    plt.plot(time, trend, label=" data average")
    plt.legend()
    plt.show()
    plt.close()

def question_b():
    residuals = labels - (time * post_mean[0] + post_mean[1])

    plt.figure()
    plt.plot(time, residuals)
    print(np.mean(residuals))
    print(np.std(residuals))

def question_c():
    cov_kernel = kernel_function(x,x)
    mean = np.zeros((cov_kernel.shape[0],))
    samples = np.random.randn(mean, cov_kernel, 100)

def kernel_function(s, t):
    cov_kernel = None
    return cov_kernel


if __name__ == "__main__":
    #question_a()
    #question_b()
    question_c()
    plt.show()

