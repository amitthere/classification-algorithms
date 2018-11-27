import numpy as np
from metrics import Metrics
from decision_tree import DecisionTree
from sklearn.model_selection import KFold


def load_d2():
    dataset2 = np.genfromtxt(r'..\data\dataset2.txt', dtype=str, delimiter='\t')

    # convert nominal attribute to numeric
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


def pa3_demo():
    return


def main():
    d1 = np.genfromtxt(r'..\data\dataset1.txt', dtype=float, delimiter='\t')
    d2 = load_d2()
    d3 = np.genfromtxt(r'..\data\project3_dataset3_train.txt', dtype=float, delimiter='\t')
    d4 = load_d4()

    # dt1 = DecisionTree().build_tree(dataset1[:-20])
    # l = DecisionTree().classify( dataset1[-3, :-1], dt1)

    # dt2 = DecisionTree().build_tree(d2[:-20])
    # l2 = DecisionTree().classify(d2[-3, :-1], dt2)

    X = d1

    # TODO Apply 10 fold cross-validation
    k=10
    kf = KFold(n_splits=k)

    accuracy = 0
    precision = 0
    recall = 0
    f1_measure = 0

    for train_index, test_index in kf.split(X):

        dt = DecisionTree()
        tree = dt.build_tree(X[train_index])
        y_pred = dt.test_classifier(X[test_index, :-1], tree)

        m = Metrics(X[test_index, -1], y_pred)
        accuracy += m.accuracy()
        precision += m.precision()
        recall += m.recall()
        f1_measure += m.f1_measure()

    # TODO Calculate Accuracy, Precision, Recall, and F-1 measure
    print('Decision Tree Accuracy ' + str(accuracy/k))
    print('Decision Tree Precision ' + str(precision/k))
    print('Decision Tree Recall ' + str(recall/k))
    print('Decision Tree F1 Measure ' + str(f1_measure/k))

    return


if __name__ == "__main__":
    main()
