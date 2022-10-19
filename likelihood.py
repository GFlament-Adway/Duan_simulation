import numpy as np

class Intensity_Model:
    def __init__(self, X, Times, Cens, Y, N=None, betas=None, eta=0, optim=None):
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
        self.n = len(X)
        self.p = len(X[0])
        self.n = len(X[0][0])
        self.C = Cens
        self.Y = Y
        self.frailty = [[eta * Y[a] for _ in range(self.n)] for a in range(len(Y))]
        print("self.Y : ", self.Y)
        if optim is None :
            self.optim = "full_like"
        else:
            self.optim = optim
        if N is None:
            self.N = np.array(Y).shape[0]
        else:
            self.N = N
        self.betas = betas
        self.eta = eta

    def f_likelihood(self, param, *args):
        """
        See equation 6.4 from D. Duffie
        args state the parameter to optimize, either beta or eta.
        fast version of likelihood.
        :return:
        """
        like = []
        likelihood = []
        if np.all(np.array(self.Y) == 0) or args[2] is False:
            """
            Case during the first step of Duffie, no need to compute all Frailty path as they are all equal.
            """
            if args[0] == "beta":
                mat = np.matmul(self.X, param).T
                mat += self.eta * np.array(args[1][0])
                intensities = np.clip(np.exp(mat), 10e-120, 10e120)
                """
                intensities = [
                    [max(min(
                        np.exp(np.sum([param[j] * self.X[k][i][j] for j in range(self.p)]) + self.eta * args[1][0][k]),
                        10e20), 10e-20) for
                     k in
                     range(int(self.T))] for
                    i in range(self.n)]
                assert np.allclose(f_intensities, intensities, atol=10e-10)
                """
            elif args[0] == "eta":
                mat = np.matmul(self.X, self.betas).T
                mat += param * np.array(args[1][0])
                intensities = np.clip(np.exp(mat), 10e-120, 10e120)
                """
                intensities = [
                    [max(min(np.exp(np.sum([self.betas[j] * self.X[k][i][j] for j in range(self.p)]) + param[0] * args[1][0][k]), 10e20), 10e-20)
                     for k in
                     range(int(self.T))] for
                    i in range(self.n)]
                assert np.allclose(f_intensities, intensities, atol=10e-10)
                """
            elif args[0] == "Y":
                mat = np.matmul(self.X, self.betas).T
                mat += self.eta * np.array(param)
                intensities = np.clip(np.exp(mat), 10e-120, 10e120)
                """    
                intensities = [
                    [max(min(np.exp(
                        np.sum([self.betas[j] * self.X[k][i][j] for j in range(self.p)]) + self.eta * param[k]),
                        10e20), 10e-20)
                        for k in
                        range(int(self.T))] for
                    i in range(self.n)]
                assert np.allclose(f_intensities, intensities, atol=10e-20)
                """
            for i in range(self.n):
                int_intensity = -np.sum(intensities[i][:int(self.Times[i])]) - intensities[i][int(self.Times[i])] * (
                        float(self.Times[i]) - int(self.Times[i]))
                assert intensities[i][int(self.Times[i])] > 0, "{inten}".format(inten=intensities[i])
                like += [(1 - self.C[i]) * (int_intensity + np.log(intensities[i][int(self.Times[i])])) + self.C[
                    i] * int_intensity]
            return -np.sum(like)

        else:
            for a in range(self.N):
                # print(a)
                log_likelihood = []
                if args[0] == "beta":
                    mat = np.matmul(self.X, param).T
                    mat += self.eta * np.array(args[1][a])
                    intensities = np.clip(np.exp(mat), 10e-120, 10e120)
                    """
                    intensities = [
                        [max(min(np.exp(np.sum([param[j] * self.X[k][i][j] for j in range(self.p)]) + self.eta[0] * args[1][a][k]), 10e20), 10e-20)
                         for k in
                         range(int(self.T))] for
                        i in range(self.n)]
                    assert np.allclose(f_intensities, intensities, atol=10e-10)
                    """
                elif args[0] == "eta":
                    mat = np.matmul(self.X, self.betas).T
                    mat += param * np.array(args[1][a])
                    intensities = np.clip(np.exp(mat), 10e-120, 10e120)
                    """
                    intensities = [
                        [max(min(np.exp(
                            np.sum([self.betas[j] * self.X[k][i][j] for j in range(self.p)]) + param[0] * args[1][a][k]), 10e20), 10e-20)
                         for k in
                         range(int(self.T))] for
                        i in range(self.n)]
                    assert np.allclose(f_intensities, intensities, atol=10e-10)
                    """
                elif args[0] == "Y":
                    mat = np.matmul(self.X, self.betas).T
                    mat += self.eta * np.array(param)
                    intensities = np.clip(np.exp(mat), 10e-120, 10e120)
                    """
                    intensities = [
                        [max(min(np.exp(
                            np.sum([self.betas[j] * self.X[k][i][j] for j in range(self.p)]) + self.eta[0] * param[k]),
                            10e20), 10e-20)
                            for k in
                            range(int(self.T))] for
                        i in range(self.n)]
                    assert np.allclose(f_intensities, intensities, atol=10e-10)
                    """
                like = []
                for i in range(self.n):
                    int_intensity = -np.sum(intensities[i][:int(self.Times[i])]) - intensities[i][int(self.Times[i])] * (
                            float(self.Times[i]) - int(self.Times[i]))
                    assert intensities[i][int(self.Times[i])] > 0, "{inten}".format(inten=intensities[i])
                    like += [(1 - self.C[i]) * (int_intensity + np.log(intensities[i][int(self.Times[i])])) + self.C[
                        i] * int_intensity]

                log_likelihood += [np.sum(like)]
            return -np.mean(log_likelihood)