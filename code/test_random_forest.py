import numpy as np
from metrics import Metrics
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


def pa3_demo(X_train, X_test):

    return


def main():
    dataset1 = np.genfromtxt(r'..\data\dataset1.txt', dtype=float, delimiter='\t')
    d3 = np.genfromtxt(r'..\data\project3_dataset3_train.txt', dtype=float, delimiter='\t')

    X = dataset1

    # TODO Apply 10 fold cross-validation
    k = 10
    kf = KFold(n_splits=k)

    for tree_count in range(5, 50, 5):
        accuracy = 0
        precision = 0
        recall = 0
        f1_measure = 0
        for train_index, test_index in kf.split(X):
            rf = RandomForest()
            forest = rf.build_forest(X[train_index], tree_count=tree_count)
            y_pred = rf.test_classifier(X[test_index, :-1], forest)

            m = Metrics(X[test_index, -1], y_pred)
            accuracy += m.accuracy()
            precision += m.precision()
            recall += m.recall()
            f1_measure += m.f1_measure()

        print('\nRandom Forest Size : ' + str(tree_count))
        print('Random Forest Accuracy ' + str(accuracy/k))
        print('Random Forest Precision ' + str(precision/k))
        print('Random Forest Recall ' + str(recall/k))
        print('Random Forest F1 Measure ' + str(f1_measure/k))

    return


if __name__ == "__main__":
    main()
