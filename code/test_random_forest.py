import numpy as np
from random_forest import RandomForest
from sklearn.model_selection import KFold


def load_d2():
    dataset2 = np.genfromtxt(r'..\data\dataset2.txt', dtype=str, delimiter='\t')

    # convert nominal attribute to continuous
    for i, v in enumerate(np.unique(dataset2[:, 4])):
        dataset2[dataset2 == v] = str(i)
    d2 = dataset2.astype('float')
    return d2


def load_d4():
    d4 = np.genfromtxt(r'..\data\project3_dataset4.txt', dtype=str, delimiter='\t')

    # convert nominal attribute to numeric
    for c in range(4):
        for i, v in enumerate(np.unique(d4[:, c])):
            d4[d4 == v] = str(i)

    dataset4 = d4.astype('float')
    return dataset4


def main():
    dataset1 = np.genfromtxt(r'..\data\dataset1.txt', dtype=float, delimiter='\t')
    dataset2 = np.genfromtxt(r'..\data\dataset2.txt', dtype=str, delimiter='\t')

    # convert nominal attribute to continuous
    for i, v in enumerate(np.unique(dataset2[:, 4])):
        dataset2[dataset2 == v] = str(i)
    d2 = dataset2.astype('float')


    # TODO Do 10-fold cross validation

    # TODO Use Metrics to calculate required numbers

    return


if __name__ == "__main__":
    main()
