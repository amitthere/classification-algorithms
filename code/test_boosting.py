import numpy as np
from metrics import Metrics
from boosting import AdaBoost
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


def pa3_demo():
    return


def main():
    d1 = np.genfromtxt(r'..\data\dataset1.txt', dtype=float, delimiter='\t')
    d2 = load_d2()
    d3 = np.genfromtxt(r'..\data\project3_dataset3_train.txt', dtype=float, delimiter='\t')
    d4 = load_d4()

    X = d1
    k = 10
    kf = KFold(n_splits=k)

    for n_classifier in range(10, 80, 10):
        accuracy = 0
        precision = 0
        recall = 0
        f1_measure = 0

        for train_index, test_index in kf.split(X):
            y_train = X[train_index, -1]
            y_train[y_train == 0] = -1

            classifier = AdaBoost(n_classifier)
            classifier.train_model(X[train_index])
            metrics = classifier.boosting_score(X[test_index, :-1], X[test_index, -1])

            accuracy += metrics.accuracy()
            precision += metrics.precision()
            recall += metrics.recall()
            f1_measure += metrics.f1_measure()

        print('\nAdaBoost with ' + str(n_classifier) + ' classifiers:')
        print('AdaBoost Accuracy ' + str(accuracy / k))
        print('AdaBoost Precision ' + str(precision / k))
        print('AdaBoost Recall ' + str(recall / k))
        print('AdaBoost F1 Measure ' + str(f1_measure / k))

    return


if __name__ == "__main__":
    main()
