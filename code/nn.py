import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import math

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

def eucli_dist(data):
    distance_matrix = euclidean_distances(data, data)
    return distance_matrix

def kfold(k, folds, data):
    rows = data.shape[0]
    accuracy = 0
    precision = 0
    recall = 0
    f1 = 0
    fold = math.ceil(rows/10)
    print("")

def metrics(ground_truth, prediction):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(ground_truth):
        if ground_truth[i] == prediction[i] == 1:
            tp += 1
        if ground_truth[i] == prediction[i] == 0:
            tn += 1
        if ground_truth[i] != prediction[i]:
            if ground_truth == 0:
                fp += 1
        if ground_truth[i] != prediction[i]:
            if ground_truth[i] == 1:
                fn += 1
    accuracy = float(tn + tp)/len(ground_truth)
    precision = float(tp)/float(tp + fp)
    recall = float(tp)/float(tp + fn)
    f1 = (2 * precision * recall)/(precision + recall)
    return accuracy, precision, recall, f1

def main():
    data, last = parse_data('../data/dataset2.txt')
    main_data = get_data(data)
    print(main_data.shape)
    main_data = main_data[:,0:-1]
    main_data = main_data.astype(float)
    #print(main_data[0:3])
    dist_mat = eucli_dist(main_data)
    #print(dist_mat[0])
    kfold(k, folds, main_data)
    return

if __name__== "__main__":
    main()