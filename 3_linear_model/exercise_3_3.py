import csv

import numpy as np
import matplotlib.pyplot as plt


def main():
    with open('../dataset/xigua_dataset_3_0_alpha.csv') as fd:
        cr = csv.reader(fd)
        rows = list(cr)[1:]

    X = np.zeros([len(rows), 2])
    y = np.zeros((len(rows),))

    for i, row in enumerate(rows):
        X[i, 0] = float(row[1])    # density
        X[i, 1] = float(row[2])    # radio sugar
        y[i] = 1 if row[3] == '是' else 0

    plt.scatter(X[y == 1, 0], X[y == 1, 1], marker='+', color='red')
    plt.scatter(X[y == 0, 0], X[y == 0, 1], marker='+', color='black')
    plt.show()


if __name__ == '__main__':
    main()
