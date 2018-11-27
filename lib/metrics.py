import numpy as np


class Metrics:

    def __init__(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = y_pred
        self.TP = 0
        self.TN = 0
        self.FP = 0
        self.FN = 0

        for i in range(len(y_true)):
            if int(y_true[i]) == int(y_pred[i]) == 1:
                self.TP += 1
            elif int(y_true[i]) == int(y_pred[i]) == 0:
                self.TN += 1
            elif int(y_true[i]) != int(y_pred[i]):
                if int(y_true[i]) == 0:
                    self.FP += 1
            elif int(y_true[i]) != int(y_pred[i]):
                if int(y_true[i]) == 1:
                    self.FN += 1

    def accuracy(self):
        _accuracy = float(self.TP + self.TN)/float(self.TP + self.TN + self.FP + self.FN)
        return _accuracy

    def precision(self):
        _precision = float(self.TP)/float(self.TP + self.FP)
        return _precision

    def recall(self):
        _recall = float(self.TP)/float(self.TP + self.FN)
        return _recall

    def f1_measure(self):
        _f1 = float(2*self.TP)/float(2*self.TP + self.FP + self.FN)
        return _f1
