from data_generator import get_data
from likelihood import Intensity_Model
import numpy as np
import matplotlib.pyplot as plt
import warnings
import seaborn as sns


def show_covariates(X):
    plt.figure()
    for i in range(len(X[0][0])):
        plt.plot(X[:, 5, i], label=r"$X_{i}$".format(i=i))
        plt.legend()
    plt.draw()


def show_true_intensity(X, true_beta, Times):
    f = []
    mat = []
    for i in range(len(true_beta)):
        mat += [np.matmul(X, [-true_beta[i][k] for k in range(len(true_beta[i]))]).T]
        f += [np.clip(np.exp(mat[i].astype(float)), 0, 10e307)]
    inds = list(np.random.choice([i for i in range(len(Times))], 2))
    plt.figure()
    for i in inds:
        for k in range(len(true_beta[0])):
            plt.plot(X[:, i, k], label=r"$X_{i}$".format(i=i))
    plt.legend()
    plt.draw()
    plt.figure()
    cols = ["r", "g", "b", "c", "m", "y", "k"]
    for i in inds:
        plt.plot([np.exp(-np.sum([f[s][i][t] for s in range(len(f))])) for t in range(len(f[0][0]))],
                 label="Conditionnal survival for ind {i}".format(i=i), color=cols[inds.index(i) % len(cols)])
        plt.axvline(x=Times[i], color=cols[inds.index(i) % len(cols)])
    plt.show()


def compare_intensity(X, true_beta, beta_hat):
    f = []
    f_hat = []
    mat = []
    mat_hat = []
    for i in range(len(true_beta)):
        mat += [np.matmul(X, [true_beta[i][k] for k in range(len(true_beta[i]))]).T]
        mat_hat += [np.matmul(X, [beta_hat[i][k] for k in range(len(beta_hat[i]))]).T]
        f += [np.clip(np.exp(mat[i].astype(float)), 0, 10e307)]
        f_hat += [np.clip(np.exp(mat_hat[i].astype(float)), 0, 10e307)]
    inds = list(np.random.choice([i for i in range(len(Times))], 1))
    plt.figure()
    cols = ["r", "g", "b", "c", "m", "y", "k"]
    for i in inds:
        plt.plot([np.exp(-np.sum([f[s][i][t] for s in range(len(f))])) for t in range(len(f[0][0]))],
                 label="Conditionnal survival for ind {i}".format(i=i), color=cols[inds.index(i) % len(cols)])
        plt.plot([np.exp(-np.sum([f_hat[s][i][t] for s in range(len(f))])) for t in range(len(f[0][0]))],
                 label="Conditionnal survival for ind {i}".format(i=i), color=cols[inds.index(i) % len(cols) + 1])
        plt.axvline(x=Times[i], color=cols[inds.index(i) % len(cols)])
    plt.show()

if __name__ == "__main__":
    warnings.filterwarnings("ignore")  # Supprime des potentiels warnings dans l'algorithme Metropolis-Hastings.
    #np.random.seed(1234)
    R = 30
    n_rep = 20
    estimated_betas = []
    n_ind = 100
    n_times = 50
    tau = 1
    all_estimated_betas = []
    best_like = []
    true_like = []
    for r in range(R):
        np.random.seed(35 * r + 10)
        print("#################################")
        print("Generating data")

        Times, X, Cens, true_beta, surv = get_data(n_ind=n_ind, n_times=n_times, tau=tau, gamma_0=20)
        print("True beta : ", true_beta)
        print("X shape : ", np.array(X).shape)
        print("mean event time : ", np.mean(Times))
        print("Censorship rate : ", np.mean(Cens))
        Y = [[0 for _ in range(len(X[0][0]))] for _ in range(len(X))]
        X = np.array([np.array(X[:, ::, i]) for i in range(len(X[0][0]))])
        print("True beta : ", true_beta)
        print("#################################")
        print("Data generated")
        #show_covariates(X)
        #show_true_intensity(X, true_beta, Times)
        intensity_model = Intensity_Model(X, Times, Cens, Y, tau=tau)
        if tau > 0:
            true_beta_like = intensity_model.f_likelihood(-np.array(true_beta[0]))
            print("Likelihood at true beta : ", intensity_model.f_likelihood(-np.array(true_beta[0])))
        else:
            true_beta_like = intensity_model.f_likelihood(np.array(true_beta))
            print("Likelihood at true beta : ", intensity_model.f_likelihood(np.array(true_beta)))
        for i in range(n_rep):
            # print(np.exp(np.matmul(X, true_beta[0]).T.astype(float)))
            # verify_theta(X, true_beta)
            # show_covariates(X)
            #   print("likelihood at true beta : ", intensity_model.f_likelihood(true_beta, to_print=True))
            test_inits = [np.random.normal(0, 1) for i in range(len(true_beta[0]))]
            if i == 0:
                beta_hat = test_inits
                print("beta hat : ", beta_hat)
                like = intensity_model.f_likelihood(beta_hat)
                print("Likelihood at initialisation : ", like)
            #   print("likelihood at initialization : ", intensity_model.f_likelihood(test_inits[-1]))
            intensity_model.fit(init=test_inits)
            current_like = intensity_model.f_likelihood(intensity_model.param[-1]["x"])
            print("Current like : ", current_like)
            if current_like < like:
                print("Found new parameters: ", intensity_model.estimated_betas)
                # print("Previous parameters was : ", beta_hat)
                print("Likelihood becomes : ", current_like, "Previously was : ", like)
                print("Number of iterations :", intensity_model.param[-1]["nit"])
                beta_hat = intensity_model.estimated_betas
                like = current_like

        #compare_intensity(X, true_beta, beta_hat)
        all_estimated_betas += [beta_hat]
        best_like += [like]
        true_like += [true_beta_like]

    plt.figure()
    plt.plot(true_like, color="b", label="Likelihood at true beta for each iteration")
    plt.plot(best_like, color="r", label="Best likelihood at each iteration")
    plt.legend()
    plt.draw()

    plt.figure()
    sns.set()
    print(all_estimated_betas)
    for k in range(tau + 1):
        print(np.array(all_estimated_betas).shape)
        box_plot = sns.boxplot([[all_estimated_betas[i][0][k][j] for i in range(R)] for j in range(len(true_beta[0]))],
                               showfliers=False)
        for xtick in box_plot.get_xticks():
            vertical_offset = np.median([[all_estimated_betas[i][0][k][xtick] for i in range(R)]]) * 0.01
            med = np.round(np.median([[all_estimated_betas[i][0][k][xtick] for i in range(R)]]), 2)
            box_plot.text(xtick, med + vertical_offset, med,
                          horizontalalignment='center', size='small', color='black', weight='semibold')
    plt.show()
