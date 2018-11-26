import numpy as np
from random_forest import RandomForest
from sklearn.model_selection import KFold


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
