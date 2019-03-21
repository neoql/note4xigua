#!/usr/bin/env python

import csv

import numpy as np
import matplotlib.pyplot as plt


def p1(x, beta):
    t = np.exp(np.dot(beta.T, x))
    return t / (1 + t)


def dl(X, y, beta):
    m = len(X)
    sum = 0
    for i in range(m):
        sum += X[i] * (y[i] - p1(X[i], beta))
    return -sum


def d2l(X, y, beta):
    m = len(X)
    sum = 0
    for i in range(m):
        x_i = X[i].reshape((3, 1))
        sum += np.dot(x_i, x_i.T) * p1(X[i], beta) * (1 - p1(X[i], beta))
    return sum


def main():
    with open('../dataset/xigua_dataset_3_0_alpha.csv') as fd:
        cr = csv.reader(fd)
        rows = list(cr)[1:]

    X = np.zeros([len(rows), 2])
    y = np.zeros((len(rows),))

    for i, row in enumerate(rows):
        X[i, 0] = float(row[1])  # density
        X[i, 1] = float(row[2])  # radio sugar
        y[i] = row[3] == '是'

    old_X = X

    d = X.shape[1]  # number of attributes
    epsilon = 0.001  # error range

    beta = np.random.rand(d + 1)
    X = np.concatenate((X, np.ones((len(X), 1))), axis=1)

    i = 0
    while True:
        i += 1
        g = dl(X, y, beta)
        if np.linalg.norm(g) < epsilon:
            break
        G = d2l(X, y, beta)
        beta = beta - np.dot(np.linalg.inv(G), g)

    X = old_X
    beta = beta.reshape(-1)

    x = np.arange(0, 0.9, 0.01)
    f = lambda x: (-beta[2] - x * beta[0]) / beta[1]

    plt.title('Logistic Regression')
    plt.xlabel('Density')
    plt.ylabel('Sugar-containing Rate')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='red', s=80, label='Good Watermelon')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='+', color='black', s=80, label='Bad Watermelon')
    plt.plot(x, f(x))
    plt.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    main()
