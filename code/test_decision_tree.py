import numpy as np
import configparser
from decision_tree import DecisionTree


def main():
    dataset1 = np.genfromtxt(r'..\data\dataset1.txt', dtype=float, delimiter='\t')
    dataset2 = np.genfromtxt(r'..\data\dataset2.txt', dtype=str, delimiter='\t')

    # convert nominal attribute to continuous
    for i, v in enumerate(np.unique(dataset2[:, 4])):
        dataset2[dataset2 == v] = str(i)
    d2 = dataset2.astype('float')

    dt1 = DecisionTree(dataset1)
    dt2 = DecisionTree(d2)
    return


if __name__ == "__main__":
    main()
