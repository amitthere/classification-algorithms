import numpy as np
from decision_tree import DecisionTree


class RandomForest:

    def __init__(self):
        pass

    def bootstrap(self, n):
        return np.random.choice(n, size=n, replace=True)

    def build_forest(self, data, tree_count, max_depth=5):

        rforest = []
        for tc in range(tree_count):
            dataset = data[self.bootstrap(len(data))]
            dt = DecisionTree()
            tree = dt.build_tree(dataset, depth=1, max_depth=max_depth, random=True)
            rforest.append(tree)
        return rforest

    def classify(self, X, rf):
        X1 = X[np.newaxis, :]

        return
