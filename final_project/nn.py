# Using https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb 
# as a template for implementing PyTorch
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

import itertools

import numpy as np

from sklearn.model_selection import KFold

import os.path
#	---	Global Statements Start	---

k = 15					# K-fold validation

epochs = 1

batch_size = 20

window_size = 2
num_classes = 8
output_size = 1

hidden_size_1 = 300
hidden_size_2 = 100

learningRate = 0.0001

input_size = num_classes * window_size 		# Input size is 7*8
#	---	Global Statements End	---

cuda = torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

class DiabetesDataset(Dataset):
	def __init__(self, batch, labels):
		self.x_data = batch
		self.y_data = labels
		self.len = self.x_data.shape[0]
		# print self.x_data
		# print self.y_data
		# print self.x_data.shape
		# print self.y_data.shape
		# print "\n\n"

	def __getitem__(self, index):
		window = torch.tensor(torch.from_numpy(self.x_data[index]))
		label = torch.tensor(self.y_data[index]).long()
		# print window.type()
		# print label
		return window, label

	def __len__(self):
		return self.len

# Neural Network Model, 1 hidden layer, relu activation function
class Net(nn.Module):
	def __init__(self, input_size, h1, h2, out_size):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(input_size, h1)	
		self.fc2 = nn.Linear(h1, h2)
		self.fc3 = nn.Linear(h2, out_size)
		self.relu = nn.ReLU()
	
	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.relu(out)
		out = self.fc3(out)
		return out
		#return F.log_softmax(out)

def train(model, epoch, data_set, criterion, optimizer, log_interval = 100):
	for batch_idx, (data, target) in enumerate(data_set):
		if cuda:
			data, target = data.cuda(), target.cuda()
		# Convert torch tensor to variable
		data, target = Variable(data), Variable(target)
		
		# forward + backward + optimize
		optimizer.zero_grad()
		output = model(data)
		loss = criterion(output, target)
		loss.backward()
		optimizer.step()
		
		if batch_idx % log_interval == 0:
			print('\t{}\t\t[{}/{} ({:.0f}%)] \t\t{:.6f}'.format(
				epoch, batch_idx * len(data), len(data_set.dataset),
				100. * batch_idx / len(data_set), loss.data.item()))
	
def validate(model, model_name, valid_set):
	#total, correct = 0, 0
	pred_name = str(model_name.split(".", 1)[0]) + "_pred.csv"
	gold_name = str(model_name.split(".", 1)[0]) + "_gold.csv"
	
	pred = open(pred_name, 'w+')
	gold = open(gold_name, 'w+')
	
	#print("Predicted\tLabel")
	for data, labels in valid_set:
		outputs = model(data)
		_, predicted = torch.max(outputs.data, 1)
		pred.write("0.1," + str(predicted.data.numpy()[0]) + "\n")
		gold.write(str(labels.data.numpy()[0]) + "\n")
		#total += labels.size(0)
		#correct += (predicted == labels).sum()
		#if predicted != labels:
			#print(str(predicted) + "\t" + str(labels))

	#print('  Correct:	%d' % correct)
	#print('    Total:	%d' % total)
	#print(' Accuracy:	%d %%' % (100 * correct / total))
	gold.close()
	pred.close()
	print("Created " + str(pred_name))
	print("Created "+ str(gold_name))
	print("Run with the following command: \n\t" + "python eval_simple.py -p " + str(pred_name) + " -g " + str(gold_name) + " -o result.csv")


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
		if  (i+window_size <= data_total_len) and (check_window(all_indice, i, i+7)):
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

def trainModel(model_name, training):
	model = Net(input_size, hidden_size_1, hidden_size_2, output_size)
	if cuda:
		model.cuda()
	criterion = nn.CrossEntropyLoss()  
	optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)  	
	print("Learning Rate: " + str(learningRate))

	for epoch in range(1, epochs + 1):
		print("\tEpoch\t\tInterval\t\tLoss")
		train(model, epoch, training, criterion, optimizer)

	torch.save(model.state_dict(), model_name)	# Saves model

def run(model_name, validation):
	model = Net(input_size, hidden_size_1, hidden_size_2, output_size)
	if cuda:
		model.cuda()
	model.load_state_dict(torch.load(model_name))
	validate(model, model_name, validation)


## Building Individual Model for Subject_2
s2_batch, s2_label = load_data('./data/part1/Subject_2_part1.csv', './data/part1/list2_part1.csv')

train_data, train_labels, test_data, test_labels = kFold(s2_batch, s2_label)

if not os.path.isfile("Subject_2.pt"):
	train_set = DiabetesDataset(train_data, train_labels,)
	train_loader = DataLoader(dataset=train_set,
							batch_size=1,
							shuffle=True,
							num_workers=2)

	print("\tPart2: Training model Subject_2.pt")
	trainModel("Subject_2.pt", train_loader)

if os.path.isfile("Subject_2.pt"):
	test_set = DiabetesDataset(test_data, test_labels)
	validation_loader = DataLoader(dataset=test_set,
							batch_size=1,
							shuffle=False,
							num_workers=2)
	print("\tPart 2: Running model Subject_2.pt")
	run("Subject_2.pt", validation_loader)

## Building Individual Model for Subject_7
s7_batch, s7_label = load_data('./data/part1/Subject_7_part1.csv', './data/part1/list_7_part1.csv')

train_data, train_labels, test_data, test_labels = kFold(s7_batch, s7_label)

if not os.path.isfile("Subject_7.pt"):
	train_set = DiabetesDataset(train_data, train_labels)
	train_loader = DataLoader(dataset=train_set,
							batch_size=1,
							shuffle=True,
							num_workers=2)

	print("\tPart2: Training model Subject_7.pt")
	trainModel("Subject_7.pt", train_loader)

if os.path.isfile("Subject_7.pt"):
	test_set = DiabetesDataset(test_data, test_labels)
	validation_loader = DataLoader(dataset=test_set,
							batch_size=1,
							shuffle=False,
							num_workers=2)
	print("\tPart 2: Running model Subject_7.pt")
	run("Subject_7.pt", validation_loader)