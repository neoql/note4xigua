import numpy as np


def cross_val_split(X, y, k=10):
    m = len(X)

    if m < k:
        raise Exception("k < len(X)")
    n = m // k
    for i in range(k):
        if i == 0:
            train_X = X[n:]
            train_y = y[n:]
            test_X = X[:n]
            test_y = y[:n]
        elif i == (k - 1):
            train_X = X[:n * (k - 1)]
            train_y = y[:n * (k - 1)]
            test_X = X[n * (k - 1):]
            test_y = y[n * (k - 1):]
        else:
            test_X = X[n * i: n * i + n]
            test_y = y[n * i: n * i + n]
            train_X = np.concatenate((X[:n * i], X[n * i + n:]))
            train_y = np.concatenate((y[:n * i], y[n * i + n:]))

        yield train_X, train_y, test_X, test_y
