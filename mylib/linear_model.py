import numpy as np


class LogisticRegression(object):
    def __init__(self):
        self.__beta = None

    def fit(self, X, y, epsilon=0.001, step_zoom=1):
        d = X.shape[1]  # number of attributes

        beta = np.random.rand(d + 1)
        X = np.concatenate((X, np.ones((len(X), 1))), axis=1)

        i = 0
        while True:
            mul = np.dot(X, beta)
            p = np.exp(mul) / (1 + np.exp(mul))
            g = -np.dot(X.T, y - p)
            if np.linalg.norm(g) < epsilon or i > 65535:
                break
            G = np.dot(X.T, (p * (1-p)).reshape((len(p), 1)) * X)
            try:
                beta = beta - np.dot(np.linalg.inv(G), g) * step_zoom
            except:
                beta = beta - np.dot(np.linalg.pinv(G), g) * step_zoom
            i += 1

        self.__beta = beta

    def predict(self, x):
        return self.__p1(np.append(x, 1), self.beta) > 0.5

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


class NotTrainException(Exception):
    def __init__(self):
        super().__init__("This learning machine has not been trained yet.")
