#!/usr/bin/env python

import csv

import numpy as np

from mylib.util import cross_val_split
from mylib.linear_model import LogisticRegression


def iris():
    np.seterr(over='ignore', invalid='ignore')

    m = 150
    dataset = np.zeros((m, 5))

    # load dataset
    with open('./dataset/iris.data.csv', 'r') as fd:
        rows = csv.reader(fd)
        for i, row in enumerate(rows):
            for j in range(4):
                dataset[i][j] = row[j]
            if row[4] == 'Iris-setosa':
                dataset[i][4] = 1
            elif row[4] == 'Iris-versicolor':
                dataset[i][4] = 2
            else:
                dataset[i][4] = 3

    np.random.shuffle(dataset)
    X = dataset[:, :4]
    y = dataset[:, 4]

    rate = [0] * 10
    k = 0
    for train_X, train_y, test_X, test_y in cross_val_split(X, y, k=10):
        # train
        cls1_X = train_X[train_y == 1]
        cls2_X = train_X[train_y == 2]
        cls3_X = train_X[train_y == 3]

        setosa_versicolor_model = LogisticRegression()
        versicolor_virginica_model = LogisticRegression()
        virginica_setosa_model = LogisticRegression()

        X1 = np.concatenate((cls1_X, cls2_X))
        X2 = np.concatenate((cls2_X, cls3_X))
        X3 = np.concatenate((cls3_X, cls1_X))

        y1 = np.concatenate((np.ones((len(cls1_X),)), np.zeros((len(cls2_X),))))
        y2 = np.concatenate((np.ones((len(cls2_X),)), np.zeros((len(cls3_X),))))
        y3 = np.concatenate((np.ones((len(cls3_X),)), np.zeros((len(cls1_X),))))

        setosa_versicolor_model.fit(X1, y1, step_zoom=0.001)
        versicolor_virginica_model.fit(X2, y2, step_zoom=0.001)
        virginica_setosa_model.fit(X3, y3, step_zoom=0.001)

        # test
        n = 0
        for i, x in enumerate(test_X):
            vote = [0, 0, 0]

            if setosa_versicolor_model.predict(x):
                vote[0] += 1
            else:
                vote[1] += 1

            if versicolor_virginica_model.predict(x):
                vote[1] += 1
            else:
                vote[2] += 1

            if virginica_setosa_model.predict(x):
                vote[2] += 1
            else:
                vote[0] += 1

            cls = vote.index(max(vote)) + 1
            if cls != test_y[i]:
                n += 1
        rate[k] = n / len(test_y)
        k += 1
    print(sum(rate) / 10)


def wine():
    dataset = np.loadtxt('./dataset/wine.data.csv', delimiter=",")


def main():
    iris()
    # wine()


if __name__ == '__main__':
    main()
