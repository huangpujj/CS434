import sys
import math
import numpy as np
import itertools
from sklearn.naive_bayes import GaussianNB
from sklearn import preprocessing
import csv

window_size = 7

def get_indice(indice = False):
    all_indice = []
    if indice is not False:
        for line in indice:
            all_indice.append(int(line))
    return all_indice

def check_window(indice, start, end):
    if len(indice) != 0:
        array = indice[start:end]
        for i, x in enumerate(array):
            if i + 1 < len(array):
                temp = x + 1
                if temp == array[i+1]:
                    continue
                else:
                    return False    # Window is not continuous
    return True                     # Window is continuous


def load_data(data_file, indice_file):
    data = np.loadtxt(data_file, delimiter=',', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9), dtype=np.float32)
    
    with open(indice_file, 'r') as f:
        indice = [line.strip() for line in f]

    data_total_len = data.shape[0]
    all_indice = get_indice(indice)

    batch = []
    labels = []

    for i, row in enumerate(data):
        new_batch = []
        if  i+window_size <= data_total_len and check_window(all_indice, i, i+7):
            for j in range(i, i+window_size):
                new_batch = [x for x in itertools.chain(new_batch, data[j, 0:8])]
                if j == i+window_size-1:
                    last = data[j, [-1]]
            batch.append(new_batch)
            labels = np.append(labels, last)
        else:
            continue            # If window size is not 7 or if the contents of the window is not continuous, skip this window

    batch = np.array(batch)
    return batch, labels


def load_test_dataV2(path):
    array = []
    with open(path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            num = 0
            for i in range(7,14,1):
                single = []
                for j in range(8):
                    single.append(float(row[i+(7*j)]))
                #label.append(float(row[i+(7*8)]))
                array.append(single)
    return np.array(array)

def load_test_data(path):
    #label = []
    array = []
    with open(path, 'rb') as csvfile:
        spamreader = csv.reader(csvfile, delimiter=',')
        for row in spamreader:
            single = []
            #num = 0
            for i in xrange(7,14,1):
                for j in range(8):
                    single.append(float(row[i+(7*j)]))
                #num += float(row[i+(7*8)])
            #if(num == 0):
                #label.append(0)
            #else:
                #label.append(1)
            array.append(single)
    return np.array(array)

def sum_data(big_batch, size):
    train = []
    for index, value in enumerate(big_batch):
        temp_matrix = [0] * 8
        for i in range(0, (size*8)):
            if(i%8 == 0):
                temp_list = []
            temp_list.append(big_batch[index][i])
            if((i+1)%8 == 0):
                #temp_matrix.append(temp_list)
                temp_matrix = [x + y for x, y in zip(temp_matrix,temp_list)]
        train.append(temp_matrix)
    return np.array(train)

def main():
    #Training Data
    s1_batch, s1_label = load_data('./data/part2/Subject_1.csv', './data/part2/list_1.csv')
    s4_batch, s4_label = load_data('./data/part2/Subject_4.csv', './data/part2/list_4.csv')
    s6_batch, s6_label = load_data('./data/part2/Subject_6.csv', './data/part2/list_6.csv')
    s9_batch, s9_label = load_data('./data/part2/Subject_9.csv', './data/part2/list_9.csv')

    in_1, in_1_label = load_data('./data/part1/Subject_2_part1.csv', './data/part1/list2_part1.csv')
    in_2, in_2_label = load_data('./data/part1/Subject_7_part1.csv', './data/part1/list_7_part1.csv')

    #Training Features
    a = np.concatenate((s1_batch, s4_batch), axis=0)
    b = np.concatenate((s6_batch, s9_batch), axis=0)
    big_batch = np.concatenate((a, b), axis=0)
    
    #Training Labels
    c = np.concatenate((s1_label, s4_label), axis=0)
    d = np.concatenate((s6_label, s9_label), axis=0)
    big_label = np.concatenate((c, d), axis=0)

    #Testing Data
    test_batch = load_test_data("data/final_test/general/general_test_instances.csv")
    in_1_test_ft = load_test_data("data/final_test/subject2/subject2_instances.csv")
    in_2_test_ft = load_test_data("data/final_test/subject7/subject7_instances.csv")
    #train = sum_data(big_batch, window_size)
    #test = sum_data(test_batch, 7)

    normal_train_group = preprocessing.normalize(big_batch, norm='l2')
    normal_test_group = preprocessing.normalize(test_batch, norm='l2')
    print test_batch[0] 
    normal_train_in_1 = preprocessing.normalize(in_1, norm='l2')
    normal_test_in_1 = preprocessing.normalize(in_1_test_ft, norm='l2')

    normal_train_in_2 = preprocessing.normalize(in_2, norm='l2')
    normal_test_in_2 = preprocessing.normalize(in_2_test_ft, norm='l2')

    model = GaussianNB()
    model_1 = GaussianNB()
    model_2 = GaussianNB()

    model.fit(normal_train_group,big_label)
    model_1.fit(normal_train_in_1,in_1_label)
    model_2.fit(normal_train_in_2,in_2_label)

    predicted = model.predict(normal_test_group)
    predict_1 = model_1.predict(normal_test_in_1)
    predict_2 = model_2.predict(normal_test_in_2)

    #scores = model.predict_proba(normal_test)
    scores = model.predict_log_proba(normal_test_group)
    scores_1 = model_1.predict_log_proba(normal_test_in_1)
    scores_2 = model_2.predict_log_proba(normal_test_in_2)
    #scores = model.score(train_data,train_label)
    #print(scores)

    predicted_int = np.array(predicted.astype(float))[np.newaxis]
    predicted_int_1 = np.array(predict_1.astype(float))[np.newaxis]
    predicted_int_2 = np.array(predict_2.astype(float))[np.newaxis]

    #Changing 'test_label' to change the gold
    #trans_goal = np.array(in_2_test_label.astype(int))[np.newaxis]
    #trans_goal = trans_goal.T

    alldata = np.append(scores,predicted_int.T,axis=1)
    alldata[:,0] = alldata[:,0] / 100.00
    alldata = alldata[:,[0,2]]

    in_1_data = np.append(scores_1,predicted_int_1.T,axis=1)
    in_1_data[:,0] = in_1_data[:,0] / 100.00
    in_1_data = in_1_data[:,[0,2]]

    in_2_data = np.append(scores_2,predicted_int_2.T,axis=1)
    in_2_data[:,0] = in_2_data[:,0] / 100.00
    in_2_data = in_2_data[:,[0,2]]

    #print alldata
    np.savetxt('results/general_pred2.csv', alldata ,delimiter=',', fmt=['%f','%d'])
    np.savetxt('results/individual1_pred2.csv', in_1_data ,delimiter=',', fmt=['%f','%d'])
    np.savetxt('results/individual2_pred2.csv', in_2_data ,delimiter=',', fmt=['%f','%d'])
    #np.savetxt('gold.csv', trans_goal ,delimiter=',', fmt='%d')

    
if __name__ == "__main__":
    main()
