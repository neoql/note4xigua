#!/usr/bin/env python

import csv

import numpy as np
import matplotlib.pyplot as plt

from mylib.linear_model import LinearDiscriminantAnalysis


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

    model = LinearDiscriminantAnalysis()
    model.fit(X, y)

    w = model.w

    e = 0
    for i, x in enumerate(X):
        label = y[i] == 1
        if model.predict(x) != label:
            e += 1
        print(model.predict(x), label)
    print(e, len(X))

    plt.title('Linear Discriminant Analysis')
    plt.xlabel('Density')
    plt.ylabel('Sugar-containing Rate')
    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='o', color='g', s=10, label='Good Watermelon')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='o', color='k', s=10, label='Bad Watermelon')
    plt.legend(loc='upper right')

    # draw line
    x = np.arange(0, 0.15, 0.01)
    fn = lambda x: w[1] / w[0] * x
    plt.plot(x, fn(x))

    for i, x in enumerate(X):
        projection = model.projection(x)
        if y[i]:
            plt.scatter(projection[0], projection[1], marker='o', color='g', s=20)
        else:
            plt.scatter(projection[0], projection[1], marker='o', color='k', s=20)
        plt.plot((x[0], projection[0]), (x[1], projection[1]), 'c--', linewidth=0.3, dashes=(10, 10))

    mu0 = model.projection(model.mu_negative)
    mu1 = model.projection(model.mu_positive)
    plt.scatter(mu0[0], mu0[1], marker='o', color='r', s=20)
    plt.scatter(mu1[0], mu1[1], marker='o', color='r', s=20)

    plt.show()


if __name__ == '__main__':
    main()
