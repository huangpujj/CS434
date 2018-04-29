import numpy as np
import pandas as pd
import sys

CANCER_VALUES = [-1.0, 1.0]

def calculate_gini(split):
    gini = float(0)
    print(split)
    for branch in split:
        print(branch)
        size = len(branch)
        if size == 0:
            continue
        for value in CANCER_VALUES:
            count = 0
            for row in branch:
                if float(row) == value:
                    count +=1
            proportion = float(count)/float(size)
            gini += (1-0)
            print(count, value)

if __name__ == "__main__":
    train_data = pd.read_csv('knn_train.csv', header=None)
    test_data = pd.read_csv('knn_test.csv', header=None)

    print(train_data.head())
    Y = train_data[[train_data.columns[0]]].values
    X = train_data.drop(train_data.columns[0], axis=1).values

    test_list = list()
    test_list.append(1)
    test_list.append(1)
    test_list.append(1)
    test_list.append(-1)
    test_list.append(-1)
    
    test_list2 = list()
    test_list2.append(1)
    test_list2.append(-1)

    big_list = test_list, test_list2
   
    
    for i in range(0, train_data.shape[1]+1):
        attribute = ([])
        attribute = np.stack((Y[:,0],X[:,i]),axis=-1)
        calculate_gini(big_list)
        sys.exit(0)
    #print(n_feature.head())
        