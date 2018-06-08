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

#	---	Global Statements Start	---

k = 10					# K-fold validation
epochs = 1
batch_size = 20
num_classes = 8
input_size = 56 		# Input size is 7*8
hidden_size_1 = 300
hidden_size_2 = 100
learningRate = 0.001

#	---	Global Statements End	---

cuda = torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

class DiabetesDataset(Dataset):
	# Retrieves all indices
	def get_indice(self, indice = False):
		self.indice = []
		if indice is not False:
			for line in open(indice):
				self.indice.append(int(line))
	
	# Checks if window[start:end] is a continuous block
	def check_window(self, start, end):
		if len(self.indice) != 0:
			array = self.indice[start:end]
			for i, x in enumerate(array):
				if i + 1 < len(array):
					temp = x + 1
					if temp == array[i+1]:
						continue
					else:
						return False	# Window is not continuous
		return True						# Window is continuous

	def __init__(self, data, indice):
		xy = np.loadtxt(data, delimiter=',', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9), dtype=np.float32)
		data_total_len = xy.shape[0]
		self.get_indice(indice)

		batch = []
		labels = []

		for i, row in enumerate(xy):
			new_batch = []
			if  i+7 <= data_total_len and self.check_window(i, i+7):
				for j in range(i, i+7):
					new_batch = [x for x in itertools.chain(new_batch, xy[j, 0:8])]
					if j == i+6:
						last = xy[j, [-1]]
				batch.append(new_batch)
				labels = np.append(labels, last)
			else:
				continue			# If window size is not 7 or if the contents of the window is not continuous, skip

		batch = np.array(batch)

		self.x_data = batch
		self.y_data = labels
		self.len = self.x_data.shape[0]
		# print self.x_data
		# print self.y_data
		#print self.x_data.shape
		#print self.y_data.shape
		#print "\n\n"

	def __getitem__(self, index):
		window = torch.tensor(torch.from_numpy(self.x_data[index]))
		label = torch.tensor(self.y_data[index]).long()
		#print window.type()
		#print label
		return window, label

	def __len__(self):
		return self.len

# Neural Network Model, 1 hidden layer, relu activation function
class Net(nn.Module):
	def __init__(self, input_size, h1, h2, num_classes):
		super(Net, self).__init__()
		self.fc1 = nn.Linear(input_size, h1)	
		self.fc2 = nn.Linear(h1, h2)
		self.fc3 = nn.Linear(h2, num_classes)
		self.relu = nn.ReLU()
	
	def forward(self, x):
		out = self.fc1(x)
		out = self.relu(out)
		out = self.fc2(out)
		out = self.relu(out)
		out = self.fc3(out)
		return out
		#return F.log_softmax(out)

def train(model, epoch, log_interval = 100):
	#model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
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
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data.item()))
	
def validate(model):
	#model.eval()
	total, correct = 0, 0
	for data, labels in validation_loader:
		outputs = model(data)
		_, predicted = torch.max(outputs.data, 1)
		total += labels.size(0)
		correct += (predicted == labels).sum()
	print('  Correct:	%d' % correct)
	print('    Total:	%d' % total)
	print(' Accuracy:	%d %%' % (100 * correct / total))


dataset = DiabetesDataset('./data/part1/Subject_2_part1.csv', './data/part1/list2_part1.csv')
train_loader = DataLoader(dataset=dataset,
						  batch_size=1,
						  shuffle=False,
						  num_workers=2)

dataset1 = DiabetesDataset('./data/part1/Subject_7_part1.csv', './data/part1/list_7_part1.csv')
validation_loader = DataLoader(dataset=dataset1,
						  batch_size=1,
						  shuffle=False,
						  num_workers=2) 

model = Net(input_size, hidden_size_1, hidden_size_2, num_classes)
if cuda:
	model.cuda()

criterion = nn.CrossEntropyLoss()  
#optimizer = optim.SGD(model.parameters(), lr=rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learningRate)  

def main():		
	print("Learning Rate: " + str(learningRate))

	for epoch in range(1, epochs + 1):
		print("\tEpoch\t\tInterval\t\tLoss")
		train(model, epoch)
		validate(model)

main()
