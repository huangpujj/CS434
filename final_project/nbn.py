import sys
import math
import numpy as np
import itertools
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

window_size = 7
k = 20

def kFold(batch, labels):
	kf = KFold(n_splits = k)
	for train_index, test_index in kf.split(batch):
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = batch[train_index], batch[test_index]
		y_train, y_test = labels[train_index], labels[test_index]

	return X_train, y_train, X_test, y_test

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

def main():
    train_file = sys.argv[1]
    train_indicies = sys.argv[2]
    batch, label = load_data(train_file,train_indicies)
    train_data, train_labels, test_data, test_labels = kFold(batch, label)
    model = GaussianNB()
    model.fit(train_data,train_labels)
    predicted = model.predict(test_data)



if __name__ == "__main__":
    main()