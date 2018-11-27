import numpy as np
import math

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
            data[0][i] = float(data[0][i])
        except:
            string_col.append(i)
    """if string_col != []:
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
    #print(data[0:10])"""
    return data, string_col

def nb(train_data, test_data, train_labels, test_labels):
    prediction = []
    return metrics(test_labels, prediction)

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
        accuracy, precision, recall, f1 = nb(train_data, test_data, train_labels, test_labels)
        totalAccuracy += accuracy
        totalPrecision += precision
        totalRecall += recall
        totalF1 += f1
    print("Accuracy after k fold validation: " ,totalAccuracy*100/folds)
    print("Precision after k fold validation: " ,totalPrecision*100/folds)
    print("Recall after k fold validation: " ,totalRecall*100/folds)
    print("F measure after k fold validation: " ,totalF1*100/folds)

def main():
    data, last = parse_data('../data/dataset2.txt')
    main_data, categorical = get_data(data)

if __name__ == "__main__":
    main()
