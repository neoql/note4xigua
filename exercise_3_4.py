#!/usr/bin/env python

import csv

import numpy as np


def iris():
    m = 150
    X = np.zeros((m, 5))
    with open('./dataset/iris.data.csv', 'r') as fd:
        dataset = csv.reader(fd)


def wine():
    dataset = np.loadtxt('./dataset/wine.data.csv', delimiter=",")


def main():
    iris()
    # wine()


if __name__ == '__main__':
    main()
