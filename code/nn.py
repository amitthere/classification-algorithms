import numpy as np

def parse_data(filepath):
    file1 = open(filepath)
    data = file1.readlines()
    #print(data)
    main_data = []
    for d in data:
        main_data.append(d.strip().split("\t"))
    #print(main_data)
    main_data = np.array(main_data)
    #print(main_data.shape)
    return main_data

def get_data(data):
    rows = data.shape[0]
    cols = data.shape[1]
    categorical_data = []
    data0 = data[0]
    string_col = []
    #print(rows, cols)
    for i in range(0,cols):
        try:
            data[0][i] == float(data[0][i])
        except:
            string_col.append(i)
    #print(string_col)
    for i in string_col:
        uniques = np.unique(data[:,i])
    #print(uniques)
    unique_dict = {}
    for i in uniques:
        unique_dict[i] = len(unique_dict)
    #print(unique_dict)
    replace_array = []
    for i in string_col:
        for j in range(0,rows):
            replace_array.append(unique_dict[data[j][i]])
        data[:,i] = replace_array
    print(data[0:10])
    return data

def main():
    data = parse_data('../data/dataset2.txt')
    main_data = get_data(data)
    return

if __name__== "__main__":
    main()