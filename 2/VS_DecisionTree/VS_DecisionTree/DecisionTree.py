import numpy as np
import pandas as pd
import sys
import pprint
import json

MAJORITY_CLASS_SIZE = 10
CANCER_VALUES = [-1.0, 1.0]

def normalize(data):
    for column in range(data.shape[1]):
        min = np.min(data[:,column])
        max = np.max(data[:,column])
        count = 0
        for row in data:
            data[count,column] = float(round((row[column] - min)/(max-min),0))
            count += 1
    return data

def calculate_gini(split):
    gini = float(0.0)
    total_sample = float(sum([len(branch) for branch in split]))
    for branch in split:
        size = len(branch)
        if size == 0:
            continue
        score =  0.0
        for value in CANCER_VALUES:
            count = 0
            for row in branch:
                if float(row[-1]) == value:
                    count +=1
            proportion = float(count)/float(size)
            score += proportion * proportion
        gini += ((1-score)*(size/total_sample))
    return gini

def split(column,value,data):
    left_branch = list()
    right_branch = list()
    for row in data:
        if row[column] < value:
            left_branch.append(row)
        else:
            right_branch.append(row)
    return left_branch,right_branch

def select_split(data):
    best_col = 1000
    best_val = 1000
    best_Left = None
    best_Right = None
    best_gini = 1000

    for column in range(len(data[0])-1):
        for row in data:
            splited = split(column, row[column], data)
            gini_val = calculate_gini(splited)
            if gini_val < best_gini:
                best_col =  column
                best_val = row[column]
                best_Left, best_Right = splited
                best_gini = gini_val
    ret_val = dict()
    ret_val['Column'] = best_col
    ret_val['Value'] = best_val
    ret_val['Left'] = best_Left
    ret_val['Right'] = best_Right
    ret_val['Gini'] = best_gini
    return ret_val


def find_Majority_Class(node):
    Y = [row[-1] for row in node]
    ones = Y.count(1)
    neg_ones = Y.count(-1)
    if neg_ones > ones:
        return neg_ones
    else:
        return ones
        

def build_Tree(curr, current_depth, max_depth):
    left_node = curr['Left']
    right_node = curr['Right']
    del(curr['Left'])
    del(curr['Right'])
    if len(left_node) == 0:
        curr['left'] = curr['right'] = find_Majority_Class(right_node)
        return
    elif len(right_node) == 0:
        curr['left'] = curr['right'] = find_Majority_Class(left_node)
        return
    if current_depth == max_depth:
        curr['left'] = find_Majority_Class(left_node)
        curr['right'] = find_Majority_Class(right_node)
        return
    if len(right_node) <= MAJORITY_CLASS_SIZE:
        curr['right'] = find_Majority_Class(right_node)
    else:
        curr['right'] = select_split(right_node)
        build_Tree(curr['right'], current_depth+1, max_depth)

    if len(left_node) <= MAJORITY_CLASS_SIZE:
        curr['left'] = find_Majority_Class(left_node)
    else:
        curr['left'] = select_split(left_node)
        build_Tree(curr['left'], current_depth+1, max_depth)

def create_Tree(data, max_depth):
    mysplit = select_split(set)
    build_Tree(mysplit, 1, max_depth)
    return mysplit
    

def printTree(final):
    print(json.dumps(final, sort_keys=True, indent=4))


def compare(final, row):
    if final["Value"] > row[final["Column"]]:
        if isinstance(final["left"], dict):
            return compare(final["left"],row)
        else:
            return final["left"]
    else:
        if isinstance(final["right"], dict):
            return compare(final["left"],row)
        else:
            return final["right"]

def getError(final, data):
    correctness = 0
    for row in data:
        check = compare(final, row)
        if check == row[0]:
            correctness +=1
    return 1 - (float(correctness))/len(data)


if __name__ == "__main__":
    if not (int(sys.argv[1]) <= 6 and int(sys.argv[1]) >= 1):
        print("Error: Usage: python DecisionTree.py <1~6>")
        sys.exit(0)
    train_data = pd.read_csv('knn_train.csv', header=None)
    test_data = pd.read_csv('knn_test.csv', header=None)

    train = train_data.values
    test = test_data.values
    Y = train_data[[train_data.columns[0]]].values
    X = train_data.drop(train_data.columns[0], axis=1).values
    
    train = normalize(train)
    test = normalize(test)
    X = normalize(X)
    set = np.append(X,Y,axis=-1)
    final = create_Tree(set, int(sys.argv[1]))
    printTree(final)
    print("Training Error: ", getError(final, train))
    print("Testing Error: ", getError(final, test))
    
   