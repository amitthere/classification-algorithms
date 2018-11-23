import numpy as np


class Node:

    def __init__(self, feature=None, left=None, right=None):
        self.feature = feature
        self.gini = None
        self.left = left
        self.right = right
        self.isLeaf = False
        self.parent = None


class DecisionTree:
    """

    """
    def __init__(self, data):
        pass

    def split(self):
        return

    def gini_impurity(self, X):
        """
        Calculate impurity of a Tree Node
        :param X: Data with labels
        :return:
        """
        counts = self.label_counts(X[:, -1])
        gini_p = 0
        for k, v in counts.items():
            gini_p += (v/len(X))**2
        return 1 - gini_p

    def label_counts(self, labels):
        """
        :param labels: Numpy array of labels
        :return: Dictionary of labels as key and their counts as value
        """
        unique, counts = np.unique(labels, return_counts=True)
        c = {}
        for i, l in enumerate(unique):
            c[l] = counts[i]
        return c

    def build_tree(self):
        tree = Node()

        return
