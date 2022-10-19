from data_generator import get_data
from likelihood import Intensity_Model
import numpy as np
import matplotlib.pyplot as plt

def show_covariates(X):
    plt.figure()
    for i in range(len(X[0][0])):
        plt.plot(X[:, 0, i], label=r"$X_{i}$".format(i=i))
        plt.plot(X[:, 3, i], label=r"$X_{i}$".format(i=i))
        plt.plot(X[:, 5, i], label=r"$X_{i}$".format(i=i))
        plt.legend()
    plt.show()

def verify_theta(X, betas):
    theta = np.matmul(X, betas)
    print(np.array(theta).shape)


if __name__ == "__main__":
    n_ind = 100
    n_times = 30

    Times, X, Cens, true_beta = get_data(n_ind=n_ind, n_times=n_times)

    Y = [[0 for _ in range(len(X[0][0]))] for _ in range(len(X))]
    X = np.array([np.array(X[:,::,i]) for i in range(len(X[0][0]))])
    verify_theta(X, true_beta)

    """
    intensity_model = Intensity_Model(X, Times, Cens, Y, tau = 0)
    print(intensity_model.betas)
    intensity_model.fit()
    print(intensity_model.param)
    """