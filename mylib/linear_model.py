import numpy as np


class LogisticRegression(object):
    def __init__(self):
        self.__beta = None

    def fit(self, X, y, epsilon=0.001, max_iter=50):
        d = X.shape[1]  # number of attributes

        beta = np.random.rand(d + 1)
        X = np.concatenate((X, np.ones((len(X), 1))), axis=1)

        i = 0
        c = 0.4
        tao = 0.4
        old_fl = self.__fn(X, y, beta)
        while True:
            step_size = 1

            mul = np.dot(X, beta)
            p = np.exp(mul) / (1 + np.exp(mul))
            isnan = np.isnan(p) | (p == 1) | (p == 0)
            if sum(isnan):
                p[isnan] = mul[isnan] / (1 + mul[isnan])
            g = -np.dot(X.T, y - p)
            if np.linalg.norm(g) < epsilon:
                break
            G = np.dot(X.T, (p * (1-p)).reshape((len(p), 1)) * X)
            try:
                pk = -np.dot(np.linalg.inv(G), g)
            except:
                pk = -np.dot(np.linalg.pinv(G), g)

            fl = self.__fn(X, y, beta + step_size * pk)

            # line search
            # Armijo criterion
            while fl > old_fl + step_size * c * np.dot(g, pk):
                step_size *= tao
                if step_size < epsilon * 0.1:
                    fl = self.__fn(X, y, beta + step_size * pk)
                    break

                fl = self.__fn(X, y, beta + step_size * pk)
            if step_size < epsilon * 0.1:
                step_size = epsilon * 0.1

            beta = beta + step_size * pk
            old_fl = fl
            i += 1
            if i > max_iter:
                break

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

    @staticmethod
    def __fn(X, y, beta):
        m = len(X)
        val = 0
        for i in range(m):
            tmp = np.dot(beta, X[i])
            v = (-y[i] * tmp + np.log(1 + np.exp(tmp)))
            if np.isinf(v) or np.isnan(v):
                v = (-y[i] * tmp) + tmp

            val += v
        return val


class NotTrainException(Exception):
    def __init__(self):
        super().__init__("This learning machine has not been trained yet.")
