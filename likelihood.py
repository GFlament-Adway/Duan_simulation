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

    def f_likelihood(self, param, s = 0):
        """
        See equation 6.4 from D. Duffie
        args state the parameter to optimize, either beta or eta.
        fast version of likelihood.
        :return:
        """
        like = []
        param = np.array(param).reshape(1, self.X.shape[2])
        mat = np.matmul(self.X, param[0]).T
        f = np.clip(np.exp(mat.astype(float)), 0, 10e307)
        assert np.count_nonzero(
            np.isnan(f)) == 0, "NaN value detected, problem with the function f, number of nan : {n}".format(
            n=np.count_nonzero(np.isnan(f)))

        for i in range(self.n):
            for t in range(self.Times[i] - s):
                if self.Times[i] > t + s + 1:
                    like += [np.exp(-f[i][t])]
                if self.Times[i] == t + s + 1 and self.C[i] == 0:
                    like += [1 - np.exp(-f[i][t])]
                if self.Times[i] == t + s + 1 and self.C[i] == 1:
                    like += [np.exp(-f[i][t])]
        return -np.sum([np.log(max(l, 10e-300)) for l in like])

    def fit(self, init=None):
        if init is None:
            init = [[np.random.normal() for _ in range(self.p)] for _ in range(self.tau + 1)]
        param = []
        for s in range(self.tau + 1):
            param += [scipy.optimize.minimize(self.f_likelihood,  x0=init, args=(s), method = "Nelder-Mead")] #method = "Nelder-mead"
        self.param = param
        if self.tau > 0:
            self.estimated_betas = [np.array([self.param[i]["x"] for i in range(self.tau + 1)])]
        else:
            self.estimated_betas = [np.array(self.param[0]["x"])]

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
