# Using Lam's code as a base to read and unpack data
import itertools
import numpy as np
from sklearn.model_selection import KFold

k = 20					# K-fold validation
batch_size = 4
window_size = 7

''' Parsing Code '''
# Retrieves all indices
def get_indice(indice = False):
	all_indice = []
	if indice is not False:
		for line in indice:
			all_indice.append(int(line))
	return all_indice

# Checks if window[start:end] is a continuous block
def check_window(indice, start, end):
	if len(indice) != 0:
		array = indice[start:end]
		for i, x in enumerate(array):
			if i + 1 < len(array):
				temp = x + 1
				if temp == array[i+1]:
					continue
				else:
					return False	# Window is not continuous
	return True						# Window is continuous

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
		if  (i+window_size <= data_total_len) and (check_window(all_indice, i, i+window_size)):
			for j in range(i, i+window_size):
				new_batch = [x for x in itertools.chain(new_batch, data[j, 0:8])]
				if j == i+window_size-1:
					last = data[j, [-1]]
			batch.append(new_batch)
			labels = np.append(labels, last)
		else:
			continue			# If window size is not 7 or if the contents of the window is not continuous, skip this window

	batch = np.array(batch)
	return batch, labels

def kFold(batch, labels):
	kf = KFold(n_splits = k)
	for train_index, test_index in kf.split(batch):
		# print("TRAIN:", train_index, "TEST:", test_index)
		X_train, X_test = batch[train_index], batch[test_index]
		y_train, y_test = labels[train_index], labels[test_index]

	return X_train, y_train, X_test, y_test

''' End of Parsing Code '''

''' Classifier Code '''
# Might actually need this to be L2 and descent?
def weight(batch, labels):
	return (np.transpose(batch).dot(labels)).dot(np.linalg.inv(np.transpose(batch).dot(batch)))   # W = (X^T * X)^-1 * X^T * Y  

# This is a problem spot, learning rate must be low or it will overflow
def sigmoid(w, f):
	return 1.0 / (1.0 + np.exp((-1.0 * np.transpose(w)).dot(f)))  # 1 / (1 + e^(-w^T x))

# Solution might be to not let g reset to 0 every time?
def gradient(w, f, o, lam = 0):
	g = np.zeros(56, dtype=float)
	for i in range(f.shape[0]):
		y_hat = sigmoid(w, f[i])                # Iterate over all features in each row
		if lam != 0:                            # If there is a lamda value then we're doing regularization for Part 2.3
			y_hat = y_hat + (lam * np.linalg.norm(w, 2))
		g = g + ((float(o[i]) - y_hat) * f[i])    # Reversed on slides, does't work for y_hat - o[i]
	return g

def batch_gradient_descent(itr, learning_rate, f_train, o_train, f_test, o_test):
	f = open("gradient_descent.csv", 'w+')
	print("Iteration\tTraining Accuracy\tTest Accuracy")
	f.write("Iteration,Training Accuracy,Test Accuracy\n")

	w = np.zeros(56, dtype=float)                      # Initilize w = [0, ...0]

	for i in range(1, itr):
		g = gradient(w, f_train, o_train)
		print("g: ")
		print(g)
		w = w + (learning_rate * g)
		print("w: ")
		print(w)
		print()
		print(str(i) + "\t" + str(check(w, f_train, o_train)) + "," + str(check(w, f_test, o_test)) + "\n")
		#f.write(str(i) + "," + str(check(w, f_train, o_train)) + "," + str(check(w, f_test, o_test)) + "\n")
	f.close()

def check(w, f, expected):  # Check predicted values agaist the correct value column and take the ratio of correct / total
	correct = 0
	for i in range(0, f.shape[0]):
		y_hat = sigmoid(w, f[i])
	if np.round(y_hat) == expected[i]:
		correct += 1
	return float(correct) / float(f.shape[0])   # Ratio expresses this weight's accuracy

def print_data(train_data, train_labels, test_data, test_labels):
	print "Training Data"
	print train_data
	print train_data.shape
	print "\nTraining Labels"
	print train_labels
	print train_labels.shape

	print "\nTest Data"
	print test_data
	print test_data.shape
	print "\nTest Labels"
	print test_labels
	print test_labels.shape
''' End Classifier Code '''

s2_batch, s2_label = load_data('./data/part1/Subject_2_part1.csv', './data/part1/list2_part1.csv')

itr = 4
learning_rate = 0.000001

train_data, train_labels, test_data, test_labels = kFold(s2_batch, s2_label)

# K, the data is reading correctly... Noting a discrepency in loaded data numbers, ~50 in the x
# print_data(train_data, train_labels, test_data, test_labels)

batch_gradient_descent(itr, learning_rate, train_data, train_labels, test_data, test_labels)
