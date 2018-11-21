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
    main_data = np.zeros(data.shape)
    categorical_data = []
    data0 = data[0]
    string_col = []
    #print(rows, cols)
    for i in range(0,cols):
        try:
            data[0][i] == float(data[0][i])
        except:
            string_col.append(i)

    print(string_col)
    return main_data

def main():
    data = parse_data('../data/dataset2.txt')
    main_data = get_data(data)
    return

if __name__== "__main__":
    main()