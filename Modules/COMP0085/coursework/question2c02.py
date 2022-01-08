import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.distributions as dist
from scipy.stats import norm

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
prior_mean = np.array([0, 360])
post_cov = np.linalg.inv(features.T @ features + np.linalg.pinv(prior_cov))
post_mean = post_cov @ (features.T @ labels + np.linalg.pinv(prior_cov)@prior_mean)



def question_a():
    print(post_cov)
    print(post_mean)
    '''
    Using least squares
    '''

    a, b = np.linalg.pinv(features.T @ features) @ (features.T @ labels)
    plt.figure()
    #plt.plot(time, a * time + b, label="least squares")

    plt.plot(time, labels, label="Data")
    plt.plot(time, time * post_mean[0] + post_mean[1], label="Posterior mean weight")
    #plt.plot(time, trend, label=" data average")
    #plt.plot(time,  time * post_mean[0] + post_mean[1] + np.random.randn(500))
    plt.xlabel("Date (decimal year)")
    plt.ylabel("$CO_2$ ppm")
    plt.legend()
    plt.show()
    plt.close()

def question_b():
    residuals = labels - (time * post_mean[0] + post_mean[1])
    mean = np.mean(residuals)
    std = np.std(residuals)
    print(mean)
    print(std)

    plt.figure()
    plt.plot(time, residuals)
    plt.xlabel("Date (decimal year)")
    plt.ylabel("Residual")

    plt.figure()
    plt.hist(residuals, density=True)
    x = np.linspace(-7, 7, 100)
    p = norm.pdf(x, mean, std)
    plt.plot(x, p, label="Normal Approximation")
    plt.ylabel("Density")
    plt.xlabel("Residual Value")
    plt.legend()

    plt.figure()
    plt.hist(residuals, density=True)
    x = np.linspace(-7, 7, 100)
    p = norm.pdf(x, mean, std)
    p_2 = norm.pdf(x, 0, 1)
    plt.plot(x, p, label="Normal Approximation")
    plt.plot(x, p_2, label="Given Approximation")
    plt.ylabel("Density")
    plt.xlabel("Residual Value")
    plt.legend()

    plt.figure()
    plt.plot(time, residuals, label="Residuals")
    plt.xlabel("Date (decimal year)")
    plt.ylabel("Residual")
    plt.plot(time, np.random.randn(500),label="Random Standard Gaussian noise")
    plt.plot(time, np.random.randn(500)*2.681)
    plt.legend()


def question_cd():

    x = np.linspace(1980, 2020, 1000)
    np.random.seed(0)

    cov_kernel = kernel_function(x, x)

    mean = np.zeros((cov_kernel.shape[0],))

    samples = np.random.randn(x.shape[0])@cov_kernel + mean

    plt.plot(x, samples)

def kernel_function(s, t):
    s = s.reshape(-1, 1)
    t = t.reshape(1, -1)

    # 3, 1, 2, 0.5, 4, 0.05
    # 2, 1, 1, 1, 3, 0.1
    theta = 3
    tau = 1
    sigma = 2
    phi = 1
    eta = 6
    zeta = 0.1

    # cov_kernel = theta^2 * (A + B) + C

    A = np.exp(-2 / (sigma**2) * (np.sin(np.pi*(s-t)/tau))**2)
    B = phi**2 * np.exp(-((s-t)**2)/(2 * eta**2))
    C = zeta**2 * (s == t)

    cov_kernel = theta**2 * (A+B) + C

    return cov_kernel

def question_ef():

    x = time
    y = labels

    features = np.concatenate((x[:, None], np.ones((x.shape))[:, None]), axis=1)
    prior_cov = np.array([[100, 0], [0, 1000]])
    post_cov = np.linalg.pinv(features.T @ features + np.linalg.pinv(prior_cov))
    post_mean = post_cov @ features.T @ y

    residuals = y - (x * post_mean[0] + post_mean[1])

    # we want to model the c02 concentration to 2030
    begin = 2021
    end = 2035
    domain = np.linspace(begin, end, 12*(end-begin))

    K = kernel_function(domain, domain)
    Kxx = kernel_function(x, x)
    Kx = kernel_function(domain, x)
    Kxx_inv = np.linalg.pinv(Kxx)

    mean = Kx @ Kxx_inv @ residuals
    Cov = K - Kx @ Kxx_inv @ Kx.T

    samples = np.random.multivariate_normal(mean, Cov, 1).T

    features_new = np.concatenate((domain[:, None], np.ones((domain.shape))[:, None]), axis=1)
    f = samples + (features_new @ post_mean)[:, None]
    f_m = mean + features_new @ post_mean

    std = np.diag(Cov).flatten()**0.5


    plt.figure()
    plt.plot(x, y, label="Data")
    plt.errorbar(domain, f_m, yerr=std, elinewidth=0.7, ecolor="darkgrey", label="Extrapolation")
    z = np.linspace(min(x), end, 1000)
    plt.plot(z, z*post_mean[0]+post_mean[1], label="MAP weights")
    plt.legend()
    plt.xlabel("Date (decimal year)")
    plt.ylabel("$CO_2$ ppm")


    plt.figure()
    z = np.linspace(begin, end, 1000)
    plt.plot(z, z * post_mean[0] + post_mean[1], label="MAP weights")
    plt.errorbar(domain, f_m, yerr=std, elinewidth=0.7, ecolor="darkgrey", label="Extrapolation")
    plt.legend()
    plt.xlabel("Date (decimal year)")
    plt.ylabel("$CO_2$ ppm")

if __name__ == "__main__":
    #question_a()
    #question_b()
    #question_cd()
    #question_ef()
    plt.show()

