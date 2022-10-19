import numpy as np
import matplotlib.pyplot as plt
import lifelines


def ornstein_uhlenbeck(t, ar):
    U = [0]
    for i in range(1, t):
        U += [ar * U[i - 1] + np.random.normal(0, 1)]
    return U


def posterior(theta_t, eta_0, gamma_0, s, t):
    post = np.exp(-(s - t) * theta_t) - np.exp(-(s - t + 1) * theta_t) * theta_t ** (eta_0 - 1) * np.exp(
        -(gamma_0 + t) * theta_t)
    return post


def get_data(n_ind=100, n_times=50, p=3, gamma_0=20, eta=1):
    """
    :param n_ind:
    :param n_times:
    :param gamma_0:
    :param eta:
    :return:
    """

    alpha = [1, 1, 1, -0.5]
    C = []
    delta = []
    Times = []
    Z = []
    p_1, p_2 = np.random.multivariate_normal([1, 1], [[0.5, 0.2], [0.2, 0.5]])
    eta_0 = 0.8

    mu = [0]

    for i in range(n_ind):
        Z_temp = []
        Z_temp += [[p_1 + p_2 * t + np.random.normal(0, 1) for t in range(n_times)]]
        Z_temp += [[t + np.random.normal() for t in range(5)] + [3 * t - 10 + np.random.normal() for t in range(5, n_times)]]
        U = ornstein_uhlenbeck(n_times, 0.5)
        for t in range(1, n_times):
            mu += [t + (mu[t - 1] * 0.2 + np.random.normal())]
        Z_temp += [[t + mu[t] for t in range(n_times)]]


        Time = np.random.gamma(gamma_0, eta)
        Z_0 = []
        for t in range(n_times):
            theta_t = 10 / (gamma_0 + t)
            Z_0 += [(np.log(theta_t) - np.sum([alpha[k] * Z_temp[k][t] for k in range(p - 1)]) - eta_0 * U[t]) / alpha[-1]]
        Z_temp += [Z_0]

        Z += [Z_temp]
        c = np.random.exponential(40)
        C += [c]
        delta += [1 if(c < Time and Time < n_times) else 0]
        Times += [min(int(Time), n_times)]
    return np.array(Times, dtype=object), np.array(Z, dtype=object), np.array(delta, dtype=object), alpha


if __name__ == "__main__":
    n_ind = 200
    U = ornstein_uhlenbeck(100, 0.4)
    Times, Z, delta = get_data(n_ind=n_ind)

    ax = plt.figure()
    ax = lifelines.plotting.plot_lifetimes(durations=Times[:20], event_observed=delta[:20], sort_by_duration=False)
    ax.set_xlabel("Event times")
    plt.show()

