import csv

import numpy as np
import matplotlib.pyplot as plt


def main():
    density = []
    ratio_sugar = []
    is_good = []

    with open('../dataset/xigua_dataset_3_0_alpha.csv') as fd:
        cr = csv.reader(fd)
        for i, row in enumerate(cr):
            if i == 0: continue
            density += [float(row[1])]
            ratio_sugar += [float(row[2])]
            is_good += [row[3] == '是']

    density = np.array(density)
    ratio_sugar = np.array(ratio_sugar)
    is_good = np.array(is_good)

    plt.scatter(density[is_good == 1], ratio_sugar[is_good == 1], marker='+', color='red')
    plt.scatter(density[is_good == 0], ratio_sugar[is_good == 0], marker='+', color='black')
    plt.show()


if __name__ == '__main__':
    main()
