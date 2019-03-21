import numpy as np


class LogisticRegression(object):
    def __init__(self):
        self.__beta = None

    def fit(self, X, y):
        dl = LogisticRegression.__dl
        d2l = LogisticRegression.__d2l

        d = X.shape[1]  # number of attributes
        epsilon = 0.001  # error range

        beta = np.random.rand(d + 1)
        X = np.concatenate((X, np.ones((len(X), 1))), axis=1)

        while True:
            g = dl(X, y, beta)
            if np.linalg.norm(g) < epsilon:
                break
            G = d2l(X, y, beta)
            beta = beta - np.dot(np.linalg.inv(G), g)

        self.__beta = beta

    def predict(self, x):
        np.append(x, 1)
        return self.__p1(x, self.beta) > 0.5

    @property
    def beta(self):
        beta = self.__beta
        if beta is None:
            raise NotTrainException()
        return beta

    @staticmethod
    def __p1(x, beta):
        t = np.exp(np.dot(beta.T, x))
        return t / (1 + t)

    @staticmethod
    def __dl(X, y, beta):
        m = len(X)
        sm = 0
        for i in range(m):
            sm += X[i] * (y[i] - LogisticRegression.__p1(X[i], beta))
        return -sm

    @staticmethod
    def __d2l(X, y, beta):
        m = len(X)
        sm = 0
        for i in range(m):
            x_i = X[i].reshape((3, 1))
            sm += np.dot(x_i, x_i.T) * LogisticRegression.__p1(X[i], beta) * (1 - LogisticRegression.__p1(X[i], beta))
        return sm


class NotTrainException(Exception):
    def __init__(self):
        super().__init__("This learning machine has not been trained yet.")
