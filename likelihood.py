import numpy as np
import scipy.optimize


class Intensity_Model:
    def __init__(self, X, Times, Cens, Y, tau = 0, N=None, eta=0, optim=None):
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
        if optim is None :
            self.optim = "full_like"
        else:
            self.optim = optim
        if N is None:
            self.N = np.array(Y).shape[0]
        else:
            self.N = N
        self.betas = [[0 for _ in range(self.p)] for _ in range(self.tau + 1)]
        self.eta = eta

    def f_likelihood(self, param, *args):
        """
        See equation 6.4 from D. Duffie
        args state the parameter to optimize, either beta or eta.
        fast version of likelihood.
        :return:
        """
        like = []
        mat = []
        exp_f = []
        param = np.array(param).reshape( self.tau + 1, len(self.betas[0]))
        for i in range(len(self.betas)):
            mat += [np.matmul(self.X, param[i]).T]
            #mat[i] += self.eta * np.array(args[0][0])
            exp_f += [np.clip(np.exp(mat[i].astype(float)), 10e-20, 10e20)]
        print("exp f : ", np.array(exp_f[0]).shape)
        for i in range(self.n):
            for t in range(self.t):
                if self.Times[i] == t + 1:
                    like += [1 - exp_f[0][i][t]]
                elif self.Times[i] <= t + self.tau + 1 and self.Times[i] >= t+1:
                    like += [1 - exp_f[int(self.Times[i]) - t - 1][i][t]]
                elif self.C[i] == 1 and self.Times[i] <= t + self.tau + 1:
                    if self.Times[i] == t + 1:
                        like += [exp_f[0][i][t]]
                    elif self.Times[i] <= t + self.tau + 1 and self.Times[i] >= t+1:
                        like += [1 - exp_f[int(self.Times[i]) - t - 1][i][t]]

        #print("likelihood contribution : ", like)
        assert len(like) == self.n, "More likelihood contribution than observatio."
        return -np.prod(like)
    def fit(self, init=None):
        if init is None:
            init = [[np.random.normal() for _ in range(self.p)] for _ in range(self.tau + 1)]
        print(init)
        param = scipy.optimize.minimize(self.f_likelihood, x0= init, args=(self.Y))
        self.param = param