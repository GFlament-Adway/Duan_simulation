from data_generator import get_data
from likelihood import Intensity_Model
import numpy as np
import matplotlib.pyplot as plt
import warnings


def show_covariates(X):
    plt.figure()
    for i in range(len(X[0][0])):
        plt.plot(X[:, 5, i], label=r"$X_{i}$".format(i=i))
        plt.legend()
    plt.draw()


if __name__ == "__main__":
    # warnings.filterwarnings("ignore")  # Supprime des potentiels warnings dans l'algorithme Metropolis-Hastings.
    np.random.seed(123456789)
    n_ind = 100
    n_times = 100
    tau = 2
    print("#################################")
    print("Generating data")

    Times, X, Cens, true_beta, surv = get_data(n_ind=n_ind, n_times=n_times, tau=tau, gamma_0=20)
    print(Times)
    print(surv)
    print("True beta : ", true_beta)
    print(np.array(X).shape)
    print("mean event time : ", np.mean(Times))
    print("Censorship rate : ", np.mean(Cens))
    Y = [[0 for _ in range(len(X[0][0]))] for _ in range(len(X))]
    X = np.array([np.array(X[:, ::, i]) for i in range(len(X[0][0]))])
    #print(np.exp(np.matmul(X, true_beta[0]).T.astype(float)))
    # verify_theta(X, true_beta)
    #show_covariates(X)
    print("True beta : ", true_beta)

    print("#################################")
    print("Data generated")
    intensity_model = Intensity_Model(X, Times, Cens, Y, tau=tau)
    print(intensity_model.betas)
    print("likelihood at true beta : ", intensity_model.f_likelihood(true_beta, to_print=True))
    sigmas = np.linspace(10e-10, 10e-1, num=100)
    test_inits = [[[np.random.normal(0, sigma) for i in range(len(true_beta[0]))] for k in range(tau + 1)] for sigma in
                  sigmas]

    print("likelihood at initialization : ", intensity_model.f_likelihood(test_inits[-1]))
    intensity_model.fit(init=test_inits[-1])
    print("beta after optim : ", intensity_model.estimated_betas)
    print("n iteration :", intensity_model.param["nit"])
    print("likelihood after optim : ", intensity_model.f_likelihood(intensity_model.param["x"]))

    preds = [intensity_model.pred(X[:, i, :], 0) for i in range(len(Times))]
    true_preds = [intensity_model.pred(X[:, i, :], 0, true_beta=true_beta) for i in range(len(Times))]
    plt.figure()
    plt.ylim(0, 1)
    for i in range(len(Times)):
        plt.plot(preds[i], color="red", alpha = 0.1)
        plt.plot(true_preds[i], color="blue", alpha = 0.1)
        #plt.plot(surv[i], color="blue", alpha = 0.1)
    plt.show()
