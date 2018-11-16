import numpy as np
import configparser
from decision_tree import DecisionTree


def main():
    dataset1 = np.genfromtxt(r'..\data\dataset1.txt', dtype=float, delimiter='\t')
    dt = DecisionTree()
    return


if __name__ == "__main__":
    main()
