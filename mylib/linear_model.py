import numpy as np


class LogisticRegression(object):
    def __init__(self):
        self.__beta = None

    def fit(self, X, y, epsilon=0.001, step_size=1):
        step_size = 1
        d = X.shape[1]  # number of attributes

        beta = np.random.rand(d + 1)
        X = np.concatenate((X, np.ones((len(X), 1))), axis=1)

        i = 0
        c = 0.4
        tao = 0.4
        old_fl = self.__fn(X, y, beta)
        while True:
            mul = np.dot(X, beta)
            p = np.exp(mul) / (1 + np.exp(mul))
            g = -np.dot(X.T, y - p)
            if np.linalg.norm(g) < epsilon:
                break
            G = np.dot(X.T, (p * (1-p)).reshape((len(p), 1)) * X)
            try:
                pk = -np.dot(np.linalg.inv(G), g)
            except:
                pk = -np.dot(np.linalg.pinv(G), g)

            fl = self.__fn(X, y, beta + step_size * pk)

            if abs(fl - old_fl) < epsilon:
                break

            while fl > old_fl + step_size * c * np.dot(g, pk):
                step_size *= tao

                fl = self.__fn(X, y, beta + step_size * pk)

            beta = beta + step_size * pk
            old_fl = fl
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

    @staticmethod
    def __fn(X, y, beta):
        m = len(X)
        val = 0
        for i in range(m):
            tmp = np.dot(beta, X[i])
            # if tmp > 700:
            #     val += (-y[i] * tmp) + tmp
            # else:
            val += (-y[i] * tmp + np.log(1 + np.exp(tmp)))
        return val


class NotTrainException(Exception):
    def __init__(self):
        super().__init__("This learning machine has not been trained yet.")
