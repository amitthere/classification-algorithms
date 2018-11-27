import numpy as np
from metrics import Metrics
from decision_tree import DecisionTree


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
    d2 = load_d2()
    d3 = np.genfromtxt(r'..\data\project3_dataset3_train.txt', dtype=float, delimiter='\t')
    d4 = load_d4()

    dt1 = DecisionTree().build_tree(dataset1[:-20])
    # l = DecisionTree().classify( dataset1[-3, :-1], dt1)
    dt2 = DecisionTree().build_tree(d2[:-20])
    # l2 = DecisionTree().classify(d2[-3, :-1], dt2)

    # TODO Apply 10 fold cross-validation

    # TODO Calculate Accuracy, Precision, Recall, and F-1 measure

    return


if __name__ == "__main__":
    main()
