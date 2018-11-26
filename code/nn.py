import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import math
from sklearn.preprocessing import StandardScaler

folds = 10
k = 5

def parse_data(filepath):
    file1 = open(filepath)
    data = file1.readlines()
    main_data = []
    for d in data:
        main_data.append(d.strip().split("\t"))
    main_data = np.array(main_data)
    last_col = main_data[:,-1]
    last_col = last_col.astype(int)
    return main_data, last_col

def get_data(data):
    rows = data.shape[0]
    cols = data.shape[1]
    string_col = []
    for i in range(0,cols):
        try:
            data[0][i] == float(data[0][i])
        except:
            string_col.append(i)
    for i in string_col:
        uniques = np.unique(data[:,i])
    unique_dict = {}
    for i in uniques:
        unique_dict[i] = len(unique_dict)
    for i in string_col:
        replace_array = []
        for j in range(0,rows):
            replace_array.append(unique_dict[data[j][i]])
        data[:,i] = replace_array
    #print(data[0:10])
    return data

def eucli_dist(train, test):
    distance_list = []
    z = len(train)
    for i in range(z):
        distance = np.linalg.norm(test-train[i])
        distance_list.append((train[i], distance, i))
    return distance_list

def knn(train, test, train_labels, test_labels):
    prediction = []
    for i in test:
        dist_matrix = eucli_dist(train, i)
        dist_matrix = sorted(dist_matrix, key = lambda x:x[1])[:k]
        one = zero = 0
        for j in dist_matrix:
            if int(train_labels[j[2]]) == 0:
                one += 1
            if int(train_labels[j[2]]) == 1:
                zero += 1
            prediction.append(1) if one > zero else prediction.append(0)
    return metrics(test_labels, prediction)


def kfold(k, folds, data):
    rows = data.shape[0]
    totalAccuracy = 0
    totalPrecision = 0
    totalRecall = 0
    totalF1 = 0
    fold = math.ceil(rows/10)
    #print(fold)
    for i in range(0, folds):
        test_data, train_data = data[i*fold:i*fold+fold,:], np.delete(data, np.s_[i*fold:i*fold+fold],0)
        test_labels = test_data[:, -1]
        train_labels = train_data[:, -1]
        train_data = train_data[:,:-1]
        test_data = test_data[:,:-1]
        norm = StandardScaler().fit(train_data)
        norm_train = norm.transform(train_data)
        norm_test = norm.transform(test_data)
        accuracy, precision, recall, f1 = knn(norm_train, norm_test, train_labels, test_labels)
        totalAccuracy += accuracy
        totalPrecision += precision
        totalRecall += recall
        totalF1 += f1
    print("Accuracy after k fold validation: " ,totalAccuracy*100/folds)
    print("Precision after k fold validation: " ,totalPrecision*100/folds)
    print("Recall after k fold validation: " ,totalRecall*100/folds)
    print("F measure after k fold validation: " ,totalF1*100/folds)

def metrics(ground_truth, prediction):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(len(ground_truth)):
        if int(ground_truth[i]) == prediction[i] == 1:
            tp += 1
        if int(ground_truth[i]) == prediction[i] == 0:
            tn += 1
        if int(ground_truth[i]) != prediction[i]:
            if int(ground_truth[i]) == 0:
                fp += 1
        if int(ground_truth[i]) != prediction[i]:
            if int(ground_truth[i]) == 1:
                fn += 1
    accuracy, precision, recall, f1 = float(tn + tp)/len(ground_truth), float(tp)/float(tp + fp), float(tp)/float(tp + fn), (2*tp)/((2*tp)+fn+fp)
    return accuracy, precision, recall, f1

def main():
    data, last = parse_data('../data/dataset2.txt')
    main_data = get_data(data)
    main_data = main_data.astype(float)
    #print(main_data[0:3])
    kfold(k, folds, main_data)

if __name__== "__main__":
    main()