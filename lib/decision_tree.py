import numpy as np


class Node:

    def __init__(self, feature=None, left=None, right=None):
        self.feature = feature
        self.gini = None
        self.parent_gini = None
        self.datapoints = None
        self.left = left
        self.right = right
        self.isLeaf = False
        self.parent = None
        self.prediction = None


class DecisionTree:
    """

    """
    def __init__(self):
        return

    def select_feature(self, data):
        maxGain = 0
        feature = None
        split = None

        features = np.array(range(data.shape[1]-1))

        for f in features:
            rsplit, gain = self.split_feature(data, f)
            if gain > maxGain:
                feature = f
                split = rsplit
                maxGain = gain

        return feature, split

    def split_feature(self, data, feature):
        split = None
        max_gain = 0
        branches = self.feature_possible_splits(data, feature)
        labels = data[:, -1]
        for fsplit in branches:
            points = self.compare(data[:, :-1], fsplit, feature)
            left_split = labels[points]
            right_split = labels[np.logical_not(points)]
            gain = self.split_gain(labels, left_split, right_split)
            if gain > max_gain:
                split = fsplit
                max_gain = gain
        return split, max_gain

    def compare(self, data, split, feature):
        return data[:, feature] <= split

    def feature_possible_splits(self, data, feature):
        limits = []
        fd = data[:, feature]
        labels = data[:, -1]
        dp = np.vstack((fd, labels)).T
        dp = dp[dp[:, 0].argsort()]
        for i in range(len(dp)):
            if i < len(dp)-1:
                if dp[i][1] != dp[i+1][1]:
                    limits.append((dp[i][0]+dp[i+1][0])/2)
        return limits

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

    def split_gain(self, parent, lchild, rchild, gini=1):
        """

        :param gini: gini index of parent's parent node
        :param parent: labels in parent node
        :param lchild: labels in Left child node
        :param rchild: labels in Right child node
        :return: Gain by using this split
        """
        gini_parent = self.gini_impurity(parent, gini)
        gini_lchild = self.gini_impurity(lchild, gini_parent)
        gini_rchild = self.gini_impurity(rchild, gini_parent)
        N, Ni, Nj = len(parent), len(lchild), len(rchild)
        gain = gini_parent - ((Ni/N)*gini_lchild + (Nj/N)*gini_rchild)
        return gain

    def majority_label(self, labels):
        label, counts = np.unique(labels, return_counts=True)
        return label[np.argmax(counts)]

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

    def build_tree(self, data, parent_gini=1):
        tree = Node()
        tree.gini = self.gini_impurity(data, tree.parent_gini)
        tree.datapoints = len(data)

        if len(np.unique(data[:, -1])) == 1:
            tree.prediction = data[0, -1]
            return tree

        split_feature, split = self.select_feature(data)
        if split_feature == None:
            tree.prediction = self.majority_label(data[:, -1])
            return tree



        return
