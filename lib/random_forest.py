import numpy as np
from decision_tree import DecisionTree


class RandomForest:

    def __init__(self):
        pass

    def classify(self, X, rf):
        X1 = X[np.newaxis, :]
        dt = DecisionTree()
        y_pred = []
        for tree in rf:
            pred = dt.classify(X1, tree)
            y_pred.append(pred)
        return dt.majority_label(y_pred)

    def test_classifier(self, X_test, rf):
        labels = []
        if X_test.ndim == 1:
            return self.classify(X_test, rf)
        for point in X_test:
            label = self.classify(point, rf)
            labels.append(label)
        return labels

    def bootstrap(self, n):
        return np.random.choice(n, size=n, replace=True)

    def build_forest(self, data, tree_count=40, max_depth=25):

        rforest = []
        for tc in range(tree_count):
            dataset = data[self.bootstrap(len(data))]
            dt = DecisionTree()
            tree = dt.build_tree(dataset, depth=1, max_depth=max_depth, random=True)
            rforest.append(tree)
        return rforest
