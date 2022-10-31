import numpy as np
import matplotlib.pyplot as plt
import lifelines
import math
from scipy.stats import halfnorm


def ornstein_uhlenbeck(t, ar):
    U = [0]
    for i in range(1, t):
        U += [ar * U[i - 1] + np.random.normal(0, 1)]
    return U


def metropolis_sampler(x_0, s, t, burn_in=50, n_sample=100, gamma_0=20, eta=1):
    """

    :param x_0:
    :param s:
    :param t:
    :param burn_in:
    :param n_sample:
    :return:
    """
    # Init step, we use gaussian distribution as the proposal density.

    x_c = np.random.normal(x_0, 1)
    x_c = np.sign(x_c) * x_c
    sample = [x_c]
    n_it = 0
    while len(sample) < 2:
        for _ in range(burn_in + n_sample):
            n_it += 1
            x_c = np.random.normal(x_0, 1)
            x_c = np.sign(x_c) * x_c
            if n_it > 1000:
                print(x_c, posterior(x_c, s, t, eta, gamma_0), x_0, posterior(x_0, s, t, eta, gamma_0))

            acceptance = posterior(x_c, s, t, eta, gamma_0) / posterior(x_0, s, t, eta, gamma_0)
            u = np.random.uniform(0, 1)
            if u <= acceptance and n_it > burn_in:
                sample += [x_c]
            else:
                x_0 = x_c
    #assert np.sum([s < 0 for s in sample]) == 0, print(np.sum([s < 0 for s in sample]))
    return sample[1:]





def posterior(theta_t, s, t, eta=1, gamma_0=20):
    post = (np.exp(-(s - t) * theta_t) - np.exp(-(s - t + 1) * theta_t)) * np.exp(
        -(gamma_0 + t) * theta_t) * theta_t ** (eta - 1)
    return post


def get_data(n_ind=100, n_times=50, gamma_0=40, eta=1, tau=0):
    """
    :param n_ind:
    :param n_times:
    :param gamma_0:
    :param eta:
    :return:
    """
    alpha = [[-5, 1] for _ in range(tau + 1)]
    C = []
    delta = []
    Times = []
    eta_0 = 0
    mu = [0]
    Z = []
    S = []

    if tau == 0:
        for i in range(n_ind):
            p_1 = np.random.normal(0, 1)
            p_2 = np.random.normal(0, 1)
            Z_temp = []
            Z_temp += [[1 if p_1 < 0 else 0 for _ in range(n_times)]]
            Z_temp += [[1 if p_2 < 0 else -1 for _ in range(n_times)]]
            """
            Z_temp += [
                [np.random.normal(-1,2) for t in range(5)] + [np.random.normal(-5, 3)
                                                                           for t in range(5, n_times)]]
            Z_temp += [
                [np.random.normal(7, 6) for t in range(5)] + [ np.random.normal(10,3)
                                                                           for t in range(5, n_times)]]
            U = ornstein_uhlenbeck(n_times, 0.5)
            for t in range(1, n_times):
                mu += [mu[t - 1] * 0.2 + np.random.normal()]
            Z_temp += [[mu[t] for t in range(n_times)]]
            """
            Z += [Z_temp]
            X = np.matmul(np.array(Z[i]).T, alpha[0])
            f = np.exp(X)
            u = np.random.uniform(0, 1)
            S += [[np.exp(-np.sum([f[:t]])) for t in range(n_times)]]
            Time = [t for t in range(len(S[i])) if S[i][t] < u or t == n_times - 1][0]
            c = np.random.exponential(50)
            C += [c]
            delta += [1 if (c < Time and Time < n_times) else 0]
            Times += [min(int(Time), n_times - 1)]
    else:
        Z = []
        p_1, p_2 = np.random.multivariate_normal([1, 1], [[0.5, 0.2], [0.2, 0.5]])
        eta_0 = 0

        for i in range(n_ind):
            Z_temp = []
            Z_temp += [[p_1 + p_2 * np.sqrt(t + 1) + np.random.normal(0, 1) for t in range(n_times)]]

            """ 
            Z_temp += [
                [np.sqrt(t + 1) + np.random.normal() for t in range(5)] + [3 * np.sqrt(t + 1) - 10 + np.random.normal()
                                                                           for t in range(5, n_times)]]
            U = ornstein_uhlenbeck(n_times, 0.5)
            for t in range(1, n_times):
                mu += [mu[t - 1] * 0.2 + np.random.normal()]
            Z_temp += [[np.log(t + 1) + mu[t] for t in range(n_times)]]
            """
            Time = np.random.gamma(gamma_0, eta)
            Z_0 = []
            for t in range(n_times):
                print("sampling : ", gamma_0)
                sample = metropolis_sampler(10 / (gamma_0 + t), Time, t, gamma_0=gamma_0, eta=eta)
                theta_t = np.random.choice(sample, 1)[0]
                print(i, Time, t, theta_t)
                assert len(Z_temp) == len(alpha[0]) - 1, "Not enough alpha"
                num = (np.log(theta_t) - np.sum(
                    np.sum([alpha[0][k] * Z_temp[k][t] for k in range(len(Z_temp))])))
                Z_0 += [num / alpha[0][-1]]
            Z_temp += [Z_0]
            Z += [Z_temp]

            c = np.random.exponential(70)
            C += [c]
            delta += [1 if (c < Time and Time < n_times) else 0]
            Times += [min(int(Time), n_times - 1)]

    return np.array(Times, dtype=object), np.array(Z, dtype=object), np.array(delta, dtype=object), alpha, S


if __name__ == "__main__":
    sample = metropolis_sampler(10, 30, 1)
    plt.figure()
    plt.hist(sample, bins=len(sample) // 20)
    plt.show()
