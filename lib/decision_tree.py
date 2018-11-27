import numpy as np


class Node:

    def __init__(self, feature=None, left=None, right=None):
        self.feature = feature
        self.gini = None
        self.split_criteria = None
        self.datapoints = None
        self.left = left
        self.right = right
        self.prediction = None


class DecisionTree:

    def __init__(self):
        return

    def classify(self, X, decision_tree):
        X1 = X[np.newaxis, :]
        if decision_tree.prediction != None:
            return decision_tree.prediction
        if self.compare(X1, decision_tree.split_criteria, decision_tree.feature):
            return self.classify(X, decision_tree.left)
        return self.classify(X, decision_tree.right)

    def test_classifier(self, data, decision_tree):
        labels = []
        if data.ndim == 1:
            return self.classify(data, decision_tree)
        for point in data:
            label = self.classify(point, decision_tree)
            labels.append(label)
        return labels

    def select_feature(self, data, random):
        """ Choose the feature among all available to be split at current node

        :param data: attribute values with label
        :return: feature to be split and value of feature at which split is done
        """
        max_gain = 0
        feature = None
        splitting_criteria = None

        if random:
            features = np.random.choice(data.shape[1]-1, int(len(data.shape[1]-1)/5), replace=False)
        else:
            features = np.array(range(data.shape[1]-1))

        for f in features:
            rsplit, gain = self.split_feature(data, f)
            if gain > max_gain:
                feature = f
                splitting_criteria = rsplit
                max_gain = gain

        return feature, splitting_criteria

    def split_feature(self, data, feature):
        """ Calculates the gain and spitting criteria for certain feature

        :param data: attribute values with label
        :param feature: used for splitting
        :return:
        """
        splitting_criteria = None
        max_gain = 0
        branches = self.feature_possible_splits(data, feature)
        labels = data[:, -1]
        for fsplit in branches:
            indices = self.compare(data[:, :-1], fsplit, feature)
            left_split = labels[indices]
            right_split = labels[np.logical_not(indices)]
            gain = self.split_gain(labels, left_split, right_split)
            if gain > max_gain:
                splitting_criteria = fsplit
                max_gain = gain
        return splitting_criteria, max_gain

    def compare(self, data, split_criteria, feature):
        return data[:, feature] <= split_criteria

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

    def gini_impurity(self, labels):
        """
        Calculate impurity of a Tree Node
        :param labels: labels in the dataset
        :return:
        """
        counts = self.label_counts(labels)
        gini_child = 0
        for k, v in counts.items():
            gini_child += (v / len(labels)) ** 2
        return 1 - gini_child

    def split_gain(self, parent, lchild, rchild):
        """

        :param parent: labels in parent node
        :param lchild: labels in Left child node
        :param rchild: labels in Right child node
        :return: Gain by using this split
        """
        gini_parent = self.gini_impurity(parent)
        gini_lchild = self.gini_impurity(lchild)
        gini_rchild = self.gini_impurity(rchild)
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

    def split_data(self, data, feature, criteria):
        indices = self.compare(data, criteria, feature)
        left = data[indices]
        right = data[np.logical_not(indices)]
        return left, right

    def build_tree(self, data, depth=1, max_depth=5, random=False):
        tree = Node()
        tree.gini = self.gini_impurity(data[:, -1])
        tree.datapoints = len(data)

        if len(np.unique(data[:, -1])) == 1 or depth == max_depth:
            tree.prediction = self.majority_label(data[:, -1])
            return tree

        split_feature, splitting_criteria = self.select_feature(data, random)
        if split_feature is None:
            tree.prediction = self.majority_label(data[:, -1])
            return tree

        tree.feature = split_feature
        tree.split_criteria = splitting_criteria

        left_data, right_data = self.split_data(data, split_feature, splitting_criteria)
        depth = depth + 1
        tree.left = self.build_tree(left_data, depth, max_depth, random)
        tree.right = self.build_tree(right_data, depth, max_depth, random)

        return tree
