import numpy as np
import sys
import csv
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
        return -1
    else:
        return 1
        

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
    return json.dumps(final, sort_keys=True, indent=4)


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


def getalldata(filename):
    all = []
    f = open(filename, 'r')
    for line in f:
        all.append(line.split(",")[0:])    # Diagnosis in this array
    f.close()
    all = np.array(all,dtype=float)
    return all


def getdata(filename):
    diagnosis = []
    features = []
    f = open(filename, 'r')
    for line in f:
        diagnosis.append(line.split(",")[0])    # Diagnosis in this array
        features.append(line.split(",")[1:])  # Features in this array
    f.close()
    (diagnosis, features) = (np.array(diagnosis, dtype=float), np.array(features, dtype=float))
    return diagnosis, features

if __name__ == "__main__":
    if not (int(sys.argv[1]) <= 6 and int(sys.argv[1]) >= 1):
        print("Error: Usage: python DecisionTree.py <1~6>")
        sys.exit(0)

    Y,X = getdata('./data/knn_train.csv')

    train = getalldata('./data/knn_train.csv')
    test = getalldata('./data/knn_test.csv')
    
    
    train = normalize(train)
    test = normalize(test)
    X = normalize(X)
    set = np.column_stack([X,Y])
    final = create_Tree(set, int(sys.argv[1]))
    tree = printTree(final)
    print(tree)
    Train_error = getError(final, train)
    Test_error = getError(final, test)
    print("Training Error: ", Train_error)
    print("Testing Error: ", Test_error)

    if int(sys.argv[1]) == 1:
        f= open("decision_stump_results.txt", 'w')
    else:
        f= open("decision_tree_results.txt", 'w')
    f.write(str(tree))
    f.write("\nTraining Error: "+str(Train_error))
    f.write("\nTesting Error: "+str(Test_error))


    
   