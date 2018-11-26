import numpy as np
from decision_tree import DecisionTree


def main():
    dataset1 = np.genfromtxt(r'..\data\dataset1.txt', dtype=float, delimiter='\t')
    dataset2 = np.genfromtxt(r'..\data\dataset2.txt', dtype=str, delimiter='\t')

    # convert nominal attribute to continuous
    for i, v in enumerate(np.unique(dataset2[:, 4])):
        dataset2[dataset2 == v] = str(i)
    d2 = dataset2.astype('float')

    dt1 = DecisionTree().build_tree(dataset1[:-20])
    # l = DecisionTree().classify( dataset1[-3, :-1], dt1)
    dt2 = DecisionTree().build_tree(d2[:-20])
    # l2 = DecisionTree().classify(d2[-3, :-1], dt2)

    # TODO Apply 10 fold cross-validation

    # TODO Calculate Accuracy, Precision, Recall, and F-1 measure

    return


if __name__ == "__main__":
    main()
