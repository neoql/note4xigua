#!/usr/bin/env python

import csv

import numpy as np
import matplotlib.pyplot as plt

from mylib.linear_model import LogisticRegression


def main():
    with open('./dataset/xigua_dataset_3_0_alpha.csv') as fd:
        cr = csv.reader(fd)
        rows = list(cr)[1:]

    X = np.zeros([len(rows), 2])
    y = np.zeros((len(rows),))

    for i, row in enumerate(rows):
        X[i, 0] = float(row[1])  # density
        X[i, 1] = float(row[2])  # radio sugar
        y[i] = row[3] == '是'

    model = LogisticRegression()
    model.fit(X, y)
    beta = model.beta

    x = np.arange(0, 0.9, 0.01)
    fn = lambda x: (-beta[2] - x * beta[0]) / beta[1]

    plt.title('Logistic Regression')
    plt.xlabel('Density')
    plt.ylabel('Sugar-containing Rate')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='red', s=80, label='Good Watermelon')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='+', color='black', s=80, label='Bad Watermelon')
    plt.plot(x, fn(x))
    plt.legend(loc='upper right')

    plt.show()


if __name__ == '__main__':
    main()
