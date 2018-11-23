import numpy as np


class Node:

    def __init__(self, feature=None, left=None, right=None):
        self.feature = feature
        self.gini = None
        self.parent_gini = None
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

    def gini_impurity(self, X, gini_parent=1):
        """
        Calculate impurity of a Tree Node
        :param X: Data with labels
        :return:
        """
        counts = self.label_counts(X[:, -1])
        gini_child = 0
        for k, v in counts.items():
            gini_child += (v/len(X))**2
        return gini_parent - gini_child

    def split_gain(self, gini, parent, lchild, rchild):
        """

        :param gini: gini index of parent's parent node
        :param parent: Data with labels in parent node
        :param lchild: Data with labels in Left child node
        :param rchild: Data with labels in Right child node
        :return: Gain by using this split
        """
        gini_parent = self.gini_impurity(parent, gini)
        gini_lchild = self.gini_impurity(lchild, gini_parent)
        gini_rchild = self.gini_impurity(rchild, gini_parent)
        N, Ni, Nj = len(parent), len(lchild), len(rchild)
        gain = gini_parent - ((Ni/N)*gini_lchild + (Nj/N)*gini_rchild)
        return gain

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
