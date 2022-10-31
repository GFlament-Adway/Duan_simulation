import math

import numpy as np
import scipy.optimize


class Intensity_Model:
    def __init__(self, X, Times, Cens, Y, tau=0, N=None, eta=0, optim=None):
        """

        :param X: Covariates
        :param Times: Event times
        :param Cens: Censure times
        :param frailty: frailty paths
        :param N: number of frailty to approximate the expectation 6.5 of D. Duffie Measuring corporate default risk.
        :param betas: parameters to be estimated
        """

        self.last_draw = 0
        self.X = X
        self.Times = Times
        self.n = len(X[0])
        self.p = len(X[0][0])
        self.t = len(X)
        self.tau = tau
        self.C = Cens
        self.Y = Y
        self.frailty = [[eta * Y[a] for _ in range(self.n)] for a in range(len(Y))]
        if optim is None:
            self.optim = "full_like"
        else:
            self.optim = optim
        if N is None:
            self.N = np.array(Y).shape[0]
        else:
            self.N = N
        self.betas = [[0 for _ in range(self.p)] for _ in range(self.tau + 1)]
        self.eta = eta

    def f_likelihood(self, param, to_print=False):
        """
        See equation 6.4 from D. Duffie
        args state the parameter to optimize, either beta or eta.
        fast version of likelihood.
        :return:
        """
        like = []
        mat = []
        f = []
        like_duffie = []
        param = np.array(param).reshape(self.tau + 1, len(self.betas[0]))
        for i in range(self.tau + 1):
            mat += [np.matmul(self.X, param[i]).T]
            # mat[i] += self.eta * np.array(args[0][0])
            f += [np.clip(np.exp(mat[i].astype(float)), 0, 10e307)]
        assert np.count_nonzero(
            np.isnan(f)) == 0, "NaN value detected, problem with the function f, number of nan : {n}".format(
            n=np.count_nonzero(np.isnan(f)))

        for i in range(self.n):
            for t in range(self.t):
                if self.C[i] == 0:
                    if self.Times[i] == t + 1:
                        like += [1 - np.exp(-f[0][i][self.Times[i]])]
                        # like += [f[0][i][self.Times[i]]]
                        # print(t, f[0][i][t])
                        # print(1 - np.exp(-f[0][i][self.Times[i]]), f[0][i][self.Times[i]])
                        assert like[-1] >= 0
                    elif self.Times[i] <= t + self.tau + 1 and self.Times[i] > t + 1:
                        like += [np.exp(-np.sum([f[l][i][t] for l in range(int(self.Times[i]) - t - 1)])) * (
                                1 - np.exp(-f[int(self.Times[i]) - t - 1][i][t]))]
                    elif self.Times[i] > t + self.tau:
                        like += [np.exp(-np.sum(f[self.tau][i][:t]))]
                elif self.C[i] == 1:
                    if self.Times[i] == t + 1:
                        like += [np.exp(-f[0][i][self.Times[i]])]
                        assert like[-1] >= 0, "time : {s}, f : {f}".format(s=self.Times[i], f=f[0][i][self.Times[i]])
                    elif self.Times[i] <= t + self.tau + 1 and self.Times[i] > t + 1:
                        #print(i, t, self.Times[i])
                        like += [np.exp(-np.sum([f[l][i][t] for l in range(int(self.Times[i]) - t - 1)])) * np.exp(
                            -f[int(self.Times[i]) - t - 1][i][t])]
                    elif self.Times[i] > t + self.tau:
                        like += [np.exp(-np.sum([f[tau][i][t] for tau in range(self.tau)]))]
                        assert like[-1] >= 0, "time : {s}, f : {f}".format(s=self.Times[i], f=[f[tau][i][t] for tau in range(self.tau)])

            like_duffie += [np.exp(-np.sum(f[0][i][:(self.Times[i])])) if self.C[i] == 1 else np.exp(
                -np.sum(f[0][i][:(self.Times[i])])) * f[0][i][self.Times[i]]]
        #print(np.sum([np.log(max(l, 10e-300)) for l in like_duffie]) - np.sum([np.log(max(l, 10e-300)) for l in like]))

        # assert len(like) == (self.tau+1)*self.n, "More likelihood contribution than observation."
        # print([np.log(min(l, 10e-324)) for l in like], np.sum(np.sum([np.log(l) for l in like])))
        assert np.array(
            [l >= 0 for l in like]).all(), "likelihood contribution negative, problem with log in index : {l}".format(
            l=np.argwhere(np.array(like) < 0))
        return -np.sum([np.log(max(l, 10e-300)) for l in like])

    def fit(self, init=None):
        if init is None:
            init = [[np.random.normal() for _ in range(self.p)] for _ in range(self.tau + 1)]
        print("Beta at initialization : ", init)
        param = scipy.optimize.minimize(self.f_likelihood, x0=init, method="Nelder-Mead")
        self.param = param
        self.estimated_betas = np.array(self.param["x"]).reshape(self.tau + 1, self.p)

    def pred(self, new_X, start_t, true_beta = None):
        if true_beta is not None:
            f_hat = []
            for t in range(self.tau + 1):
                f_hat += [np.exp(-np.matmul(new_X, true_beta[t]).T.astype(float))]
            p = []
            for t in range(start_t, len(f_hat[0])):
                p += [[np.exp(-np.sum(f_hat[i][start_t:t])) for i in range(self.tau + 1)]]
            return p
        else:
            f_hat = []
            for t in range(self.tau + 1):
                f_hat += [np.exp(np.matmul(new_X, self.estimated_betas[t]).T.astype(float))]
            p = []
            for t in range(start_t, len(f_hat[0])):
                p += [[np.exp(-np.sum(f_hat[i][start_t:t])) for i in range(self.tau + 1)]]
            return p
