import numpy as np
import math
import pandas as pd

string1_prob1 = 0
string1_prob2 = 0
string2_prob1 = 0
string2_prob2 = 0

def parse_data(filepath):
    file = pd.read_csv(filepath, sep = '\t', header = None)
    main_data = file.as_matrix()
    return main_data

def calc_std(main_data, data, mean, strings):
    deviations = []
    if strings == []:
        for i in range(0, main_data.shape[1]-1):
            dev = math.sqrt(np.sum([np.power(k-mean[i],2) for k in data[:,i]])/len(data))
            deviations.append(dev)
    else:
        for i in range(0,strings[0]):
            dev = math.sqrt(np.sum([np.power(k - mean[i], 2) for k in data[:, i]]) / len(data))
            deviations.append(dev)
        deviations.append(1)
        for i in range(strings[0]+1, main_data.shape[1]-1):
            dev = math.sqrt(np.sum([np.power(k - mean[i], 2) for k in data[:, i]]) / len(data))
            deviations.append(dev)
    return deviations

def post_prob(main_data, mean, std_dev, data, strings):
    post = []
    if strings == []:
        for i in range(0, len(data)-1):
            e = np.exp(-np.power((data[i]-mean[i]),2)/(2*np.power(std_dev[i],2)))
            pd = (1/(np.sqrt(2*np.pi)*std_dev[i])) * e
            post.append(pd)
    else:
        for i in range(0, strings[0]):
            e = np.exp(-np.power((data[i] - mean[i]), 2) / (2 * np.power(std_dev[i], 2)))
            pd = (1 / (np.sqrt(2 * np.pi) * std_dev[i])) * e
            post.append(pd)
        post.append(1)
        for i in range(strings[0]+1, main_data.shape[1]-1):
            e = np.exp(-np.power((data[i] - mean[i]), 2) / (2 * np.power(std_dev[i], 2)))
            pd = (1 / (np.sqrt(2 * np.pi) * std_dev[i])) * e
            post.append(pd)
    return post


def nb(main_data, train_data, test_data, test_labels, strings):
    prediction = []
    rows = main_data.shape[0]
    cols = main_data.shape[1]
    class0 = []
    class1 = []
    for i in range(0,len(train_data)):
        class1.append(train_data[i]) if train_data[i][cols-1] == 1 else class0.append(train_data[i])
    class0 = np.array(class0)
    class1 = np.array(class1)
    c0row = class0.shape[0]
    c1row = class1.shape[0]
    trshape = train_data.shape[0]
    means0 = []
    means1 = []
    if strings == []:
        for i in range(0,main_data.shape[1]-1):
            mean0 = np.mean(class0[:,i])
            mean1 = np.mean(class1[:,i])
            means0.append(mean0)
            means1.append(mean1)
    else:
        for i in range(0, strings[0]):
            mean0 = np.mean(class0[:, i])
            mean1 = np.mean(class1[:, i])
            means0.append(mean0)
            means1.append(mean1)
        means0.append(1)
        means1.append(1)
        for i in range(strings[0]+1, main_data.shape[1]-1):
            mean0 = np.mean(class0[:, i])
            mean1 = np.mean(class1[:, i])
            means0.append(mean0)
            means1.append(mean1)
    #print(len(means0), len(means1))
    std_dev0 = calc_std(main_data, class0, means0, strings)
    std_dev1 = calc_std(main_data, class1, means1, strings)
    #print(len(std_dev0), len(std_dev1))
    cat_data = []
    a = b = c = d = 0
    if strings != []:
        for j in strings:
            for i in range(0,len(class0)):
                if class0[i][j] not in cat_data:
                    cat_data.append(class0[i][j])
        for j in strings:
            for i in range(0, len(class0)):
                if class0[i,j] == cat_data[1]:
                    a += 1
                elif class0[i,j] == cat_data[0]:
                    b += 1
            string1_prob1 = a/(len(class0))
            string2_prob1 = b/(len(class0))
            for i in range(0, len(class1)):
                if class1[i,j] == cat_data[1]:
                    c += 1
                elif class1[i,j] == cat_data[0]:
                    d += 1
            string1_prob2 = c/(len(class1))
            string2_prob2 = d/(len(class1))
    #print(string2_prob2)
    test_label = []
    ground_truth = []
    pp0 = c0row/trshape
    pp1 = c1row/trshape
    for i in range(0, len(test_data)):
        ground_truth.append(test_data[i][cols-1])
        post_prob1 = post_prob(main_data, means0, std_dev0, test_data[i], strings)
        post_prob2 = post_prob(main_data, means1, std_dev1, test_data[i], strings)
        post1 = np.prod(post_prob1, axis = None)
        post2 = np.prod(post_prob2, axis = None)
        if strings != []:
            if test_data[i][strings[0]] == cat_data[1]:
                class_post1 = post1 * pp0 * string1_prob1
                class_post2 = post2 * pp1 * string1_prob2
            elif test_data[i][strings[0]] == cat_data[0]:
                class_post1 = post1 * pp0 * string2_prob1
                class_post2 = post2 * pp1 * string2_prob2
        else:
            class_post1 = post1 * pp0
            class_post2 = post2 * pp1
        test_label.append(0) if class_post1 > class_post2 else test_label.append(1)
    return metrics(test_label, ground_truth)

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

def kfold(k, data, string_col):
    rows = data.shape[0]
    cols = data.shape[1]
    totalAccuracy = 0
    totalPrecision = 0
    totalRecall = 0
    totalF1 = 0
    fold = math.ceil(data.shape[0]/10)
    #print(fold)
    for i in range(0, k):
        test_data =  data[i*fold:i*fold+fold,:]
        train_data =  np.delete(data, np.s_[i*fold:i*fold + fold],0)
        test_labels = data[fold*(i-1):fold*i,-1]
        accuracy, recall, precision, f1 = nb(data, train_data, test_data, test_labels, string_col)
        totalAccuracy += accuracy
        totalPrecision += precision
        totalRecall += recall
        totalF1 += f1
    print("Accuracy after k fold validation: " ,totalAccuracy*100/fold)
    print("Precision after k fold validation: " ,totalPrecision*100/fold)
    print("Recall after k fold validation: " ,totalRecall*100/fold)
    print("F measure after k fold validation: " ,totalF1*100/fold)

def main():
    main_data = parse_data('../data/dataset2.txt')
    string_col = []
    for i in range(0,main_data.shape[1]):
        try:
            main_data[0][i] == float(main_data[0][i])
        except:
            string_col.append(i)
    #print(string_col)
    kfold(10,main_data, string_col)

if __name__ == "__main__":
    main()
