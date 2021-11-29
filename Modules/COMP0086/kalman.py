# -*- coding: utf-8 -*-

"""
    File name: ssm_kalman.py
    Description: a re-implementation of the Kalman filter for http://www.gatsby.ucl.ac.uk/teaching/courses/ml1
    Author: Roman Pogodin / Maneesh Sahani (matlab version)
    Date created: October 2018
    Python version: 3.6
"""

import numpy as np
import os
import matplotlib.pyplot as plt


def run_ssm_kalman(X, y_init, Q_init, A, Q, C, R, mode='smooth'):
    """
    Calculates kalman-smoother estimates of SSM state posterior.
    :param X:       data, [d, t_max] numpy array
    :param y_init:  initial latent state, [k,] numpy array
    :param Q_init:  initial variance, [k, k] numpy array
    :param A:       latent dynamics matrix, [k, k] numpy array
    :param Q:       innovariations covariance matrix, [k, k] numpy array
    :param C:       output loading matrix, [d, k] numpy array
    :param R:       output noise matrix, [d, d] numpy array
    :param mode:    'forw' or 'filt' for forward filtering, 'smooth' for also backward filtering
    :return:
    y_hat:      posterior mean estimates, [k, t_max] numpy array
    V_hat:      posterior variances on y_t, [t_max, k, k] numpy array
    V_joint:    posterior covariances between y_{t+1}, y_t, [t_max, k, k] numpy array
    likelihood: conditional log-likelihoods log(p(x_t|x_{1:t-1})), [t_max,] numpy array
    """
    d, k = C.shape
    t_max = X.shape[1]

    # dimension checks
    assert np.all(X.shape == (d, t_max)), "Shape of X must be (%d, %d), %s provided" % (d, t_max, X.shape)
    assert np.all(y_init.shape == (k,)), "Shape of y_init must be (%d,), %s provided" % (k, y_init.shape)
    assert np.all(Q_init.shape == (k, k)), "Shape of Q_init must be (%d, %d), %s provided" % (k, k, Q_init.shape)
    assert np.all(A.shape == (k, k)), "Shape of A must be (%d, %d), %s provided" % (k, k, A.shape)
    assert np.all(Q.shape == (k, k)), "Shape of Q must be (%d, %d), %s provided" % (k, k, Q.shape)
    assert np.all(C.shape == (d, k)), "Shape of C must be (%d, %d), %s provided" % (d, k, C.shape)
    assert np.all(R.shape == (d, d)), "Shape of R must be (%d, %d), %s provided" % (d, k, R.shape)

    y_filt = np.zeros((k, t_max))  # filtering estimate: \hat(y)_t^t
    V_filt = np.zeros((t_max, k, k))  # filtering variance: \hat(V)_t^t
    y_hat = np.zeros((k, t_max))  # smoothing estimate: \hat(y)_t^T
    V_hat = np.zeros((t_max, k, k))  # smoothing variance: \hat(V)_t^T
    K = np.zeros((t_max, k, X.shape[0]))  # Kalman gain
    J = np.zeros((t_max, k, k))  # smoothing gain
    likelihood = np.zeros(t_max)  # conditional log-likelihood: p(x_t|x_{1:t-1})

    I_k = np.eye(k)

    # forward pass

    V_pred = Q_init
    y_pred = y_init

    for t in range(t_max):
        x_pred_err = X[:, t] - C.dot(y_pred)
        V_x_pred = C.dot(V_pred.dot(C.T)) + R
        V_x_pred_inv = np.linalg.inv(V_x_pred)
        likelihood[t] = -0.5 * (np.linalg.slogdet(2 * np.pi * (V_x_pred))[1] +
                                x_pred_err.T.dot(V_x_pred_inv).dot(x_pred_err))

        K[t] = V_pred.dot(C.T).dot(V_x_pred_inv)

        y_filt[:, t] = y_pred + K[t].dot(x_pred_err)
        V_filt[t] = V_pred - K[t].dot(C).dot(V_pred)

        # symmetrise the variance to avoid numerical drift
        V_filt[t] = (V_filt[t] + V_filt[t].T) / 2.0

        y_pred = A.dot(y_filt[:, t])
        V_pred = A.dot(V_filt[t]).dot(A.T) + Q

    # backward pass

    if mode == 'filt' or mode == 'forw':
        # skip if filtering/forward pass only
        y_hat = y_filt
        V_hat = V_filt
        V_joint = None
    else:
        V_joint = np.zeros_like(V_filt)
        y_hat[:, -1] = y_filt[:, -1]
        V_hat[-1] = V_filt[-1]

        for t in range(t_max - 2, -1, -1):
            J[t] = V_filt[t].dot(A.T).dot(np.linalg.inv(A.dot(V_filt[t]).dot(A.T) + Q))
            y_hat[:, t] = y_filt[:, t] + J[t].dot((y_hat[:, t + 1] - A.dot(y_filt[:, t])))
            V_hat[t] = V_filt[t] + J[t].dot(V_hat[t + 1] - A.dot(V_filt[t]).dot(A.T) - Q).dot(J[t].T)

        V_joint[-2] = (I_k - K[-1].dot(C)).dot(A).dot(V_filt[-2])

        for t in range(t_max - 3, -1, -1):
            V_joint[t] = V_filt[t + 1].dot(J[t].T) + J[t + 1].dot(V_joint[t + 1] - A.dot(V_filt[t + 1])).dot(J[t].T)

    return y_hat, V_hat, V_joint, likelihood

def M_step(X, y_hat, V_hat, V_joint):
    """
    M-step for the EM algorithm
    """
    # we follow the equations in the notes and in the question to create the matrices A,C,Q and R
    # We note that the V and Vj are variances
    t_max = X.shape[1]
    A_new = (np.sum(V_joint[:999, :, :], axis=0) + y_hat[:, 1:] @ y_hat[:, :999].T) @ np.linalg.inv(np.sum(V_hat, axis=0) + y_hat @ y_hat.T)
    C_new = X @ y_hat.T @ np.linalg.inv(np.sum(V_hat, axis=0) + y_hat @ y_hat.T)
    R_new = 1/t_max * (X@X.T - X@y_hat.T @ C_new.T)
    Q_new = 1/(t_max-1) * ((np.sum(V_hat[1:, :, :], axis=0) + y_hat[:, 1:] @ y_hat[:, 1:].T
                           - (np.sum(V_joint[:999, :, :], axis=0) + y_hat[:, 1:] @ y_hat[:, :999].T)  @ A_new.T))

    return A_new, C_new, R_new, Q_new

def EM_initial(N,X, X_test):

    """
    We define an EM loop that uses the given inital parameters by the question
    """
    theta = 2 * np.pi / 180
    A = 0.99 * np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                         [np.sin(theta), np.cos(theta), 0, 0],
                         [0, 0, np.cos(2 * theta), -np.sin(2 * theta)],
                         [0, 0, np.sin(2 * theta), np.cos(2 * theta)]])

    C = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 0, 0, 1],
                  [0, 0, 1, 1],
                  [0.5, 0.5, 0.5, 0.5]])

    Q = np.eye(4) - A @ A.T
    R = np.eye(5)
    Y0 = np.zeros((4,))
    Q0 = np.eye(4)

    likelihoods = []
    for n in range(N):
        Y, V, Vj, L = run_ssm_kalman(X, Y0, Q0, A, Q, C, R, 'smooth')
        A, C, R, Q = M_step(X, Y, V, Vj)
        likelihoods.append(np.sum(L))

    # we use the parameters found to find the likelihood on the test set
    Y_, V_, Vj_, L_test = run_ssm_kalman(X_test, Y0, Q0, A, Q, C, R, 'smooth')
    return likelihoods, np.sum(L_test)


def EM_random(N,X, X_test):
    """
    We define an EM loop that uses random parameters to initialise
    """
    # certain seeds work better than others and give nicer plots
    #np.random.seed(144) # bad: 504,54,1,2 good: 3,10, 144
    A = np.random.rand(4, 4)

    C = np.random.rand(5, 4)

    Q = np.eye(4) - A @ A.T
    R = np.eye(5)
    Y0 = np.random.randn(4,)
    Q0 = np.eye(4)


    likelihoods = []
    for n in range(N):
        Y, V, Vj, L = run_ssm_kalman(X, Y0, Q0, A, Q, C, R, 'smooth')
        A, C, R, Q = M_step(X, Y, V, Vj)
        likelihoods.append(np.sum(L))
    # we use the parameters found to find the likelihood on the test set
    Y_, V_, Vj_, L_test = run_ssm_kalman(X_test, Y0, Q0, A, Q, C, R, 'smooth')
    return likelihoods, np.sum(L_test)

def plot_q_a(X):
    """
    function to plot the plots requested in part a
    """
    theta = 2 * np.pi / 180
    A = 0.99 * np.array([[np.cos(theta), -np.sin(theta), 0, 0],
                         [np.sin(theta), np.cos(theta), 0, 0],
                         [0, 0, np.cos(2 * theta), -np.sin(2 * theta)],
                         [0, 0, np.sin(2 * theta), np.cos(2 * theta)]])

    C = np.array([[1, 0, 1, 0],
                  [0, 1, 0, 1],
                  [1, 0, 0, 1],
                  [0, 0, 1, 1],
                  [0.5, 0.5, 0.5, 0.5]])

    Q = np.eye(4) - A @ A.T
    R = np.eye(5)
    Y0 = np.zeros((4,))
    Q0 = np.eye(4)
    logdet = lambda x: 2 * np.sum(np.log(np.diag(np.linalg.cholesky(x))))
    Y, V, Vj, L = run_ssm_kalman(X, Y0, Q0, A, Q, C, R, 'filt')
    plt.figure()
    plt.plot(Y.T)
    vector = [logdet(V_elements) for V_elements in V]
    plt.show()
    plt.figure()
    plt.plot(vector)
    plt.show()
    Y, V, Vj, L = run_ssm_kalman(X,Y0,Q0,A,Q,C,R,'smooth')
    plt.figure()
    plt.plot(Y.T)
    vector = [logdet(V_elements) for V_elements in V]
    plt.show()
    plt.figure()
    plt.plot(vector)
    plt.show()

N = 50
seed = [3, 10, 144, 50, 34] # seeds that work well (not used)
X = np.loadtxt("http://www.gatsby.ucl.ac.uk/teaching/courses/ml1/ssm_spins.txt").T
X_test = np.loadtxt("http://www.gatsby.ucl.ac.uk/teaching/courses/ml1/ssm_spins_test.txt").T


if __name__ == "__main__":
    plot_q_a(X)

    # for part b we want to run the EM algorithm multiple times for different initial conditions
    # for part c we want to compare the final likelihood of the training set and use those parameters on the test set.
    test_set = []

    # we save the final likelihoods and test likelihoods to make a table
    likes = np.zeros((11,2))

    # True parameters first
    L, L_test = EM_initial(N, X, X_test)
    likes[10,:] = L[-1], L_test
    plt.figure()

    plt.plot(L, label="Given Initials")

    test_set.append(L_test)

    # we plot the likelihood of the test set as a point on the right of the plot for easier comparison
    plt.plot(N-1, L_test,".", label="Test Set with True")

    for n in range(10):
        L_r, L_test = EM_random(N,X,X_test)
        plt.plot(L_r)
        test_set.append(L_test)
        likes[n, :] = L_r[-1], L_test

    plt.xlabel("Iterations")
    plt.ylabel("Log-Likelihood")

    # we plot the likelihood of the test set as a point on the right of the plot for easier comparison
    plt.plot(N*np.ones((len(likes[:, 1]))), [i for i in likes[:, 1]], ".", label="Test set with random")
    plt.grid()
    plt.legend()
    plt.show()
