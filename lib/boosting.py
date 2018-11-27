import numpy as np
from metrics import Metrics
from decision_tree import DecisionTree


class AdaBoost:

    def __init__(self, n_classifier):
        self.n_classifier = n_classifier
        self.classifiers = []
        self.classifier_weight = []

    def bootstrap(self, weights):
        return np.random.choice(len(weights), size=len(weights), replace=True, p=weights)

    def train_model(self, X, y):
        weights = np.ones(len(X))/len(X)

        while self.n_classifier > len(self.classifiers):
            sample_indices = self.bootstrap(weights)
            X_train = X[sample_indices]
            y_train = y[sample_indices]

            data = np.vstack((X_train, y_train))
            dt = DecisionTree()
            tree = dt.build_tree(data, max_depth=5)
            y_pred = dt.test_classifier(X, tree)

            error = weights.dot(y_pred != y)
            if error < 0.5:
                alpha = (np.log(1 - error) - np.log(error))/2.0
                weights = weights*np.exp(-alpha*y*y_pred)
                weights = weights/sum(weights)
                self.classifiers.append(tree)
                self.classifier_weight.append(alpha)

    def classify(self, X_test):
        yp = np.zeros(len(X_test))
        dt = DecisionTree()
        for i in range(len(self.classifiers)):
            yp = yp + self.classifier_weight[i]*np.array(dt.test_classifier(X_test, self.classifiers[i]))
        return np.sign(yp)

    def boosting_score(self, X, y):
        y_pred = self.classify(X)
        y_pred[y_pred == -1] = 0
        return Metrics(y, y_pred)
