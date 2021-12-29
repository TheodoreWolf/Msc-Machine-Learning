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

def question_cd():

    def naive_GP():
        cov_kernel = kernel_function(time, time)
        mean = np.zeros((cov_kernel.shape[0],))
        samples = np.random.multivariate_normal(mean, cov_kernel, 10).T

        plt.plot(time, samples)

    def post_GP():

        # y = np.array([labels[10], labels[40], labels[46], labels[305], labels[406]])
        # x = np.array([time[10], time[40], time[46], time[305], time[406]])
        x = time
        y = labels
        domain = np.linspace(1950, 2050, 1000)
        K = kernel_function(domain, domain)
        Kxx = kernel_function(x, x)
        Kx = kernel_function(domain, x)
        Kxx_inv = np.linalg.pinv(Kxx)

        mean = Kx @ Kxx_inv @ y
        Cov = K - Kx @ Kxx_inv @ Kx.T

        samples = np.random.multivariate_normal(mean, Cov, 1).T

        std = np.diag(Cov)**0.5
        plt.fill_between(domain, mean - std, mean + std, color='k', alpha=0.1)
        plt.fill_between(domain, mean - 2 * std, mean + 2 * std, color='k', alpha=0.1)
        plt.plot(domain, samples)

    post_GP()
    #plt.plot(time, labels)

def kernel_function(s, t):
    s = s.reshape(-1, 1)
    t = t.reshape(1, -1)
    theta = 1
    tau = 1
    sigma = 1
    phi = 1
    eta = 1
    zeta = 0.01

    # cov_kernel = theta^2(A + B) +C

    A = np.exp(-2 / (sigma**2) * (np.sin(np.pi*(s-t)/tau))**2)
    B = phi**2 * np.exp(-((s-t)**2)/(2 * eta**2))
    C = zeta**2 * (s == t)

    cov_kernel = theta**2 * (A+B) + C

    return cov_kernel


if __name__ == "__main__":
    #question_a()
    #question_b()
    question_cd()
    plt.show()

