# Using https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb 
# as a template for implementing PyTorch
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable	# Deprecated
from torch.utils.data import Dataset, DataLoader

import itertools

import numpy as np

# --- Global Statements Start ---
learningRates = [0.1]
epochs = 1
batch_size = 32

if (len(sys.argv) > 1):
	DROPOUT = float(sys.argv[1])
	MOMENTUM = float(sys.argv[2])
	WEIGHT_DECAY = float(sys.argv[3])
else:
	DROPOUT = float(0.0)
	MOMENTUM = float(0.0)
	WEIGHT_DECAY = float(0.0)

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
		
		self.len = xy.shape[0]

		self.get_indice(indice)

		batch = []
		labels = []

		for i, row in enumerate(xy):
			new_batch = []
			if  i+7 <= self.len and self.check_window(i, i+7):
				for j in range(i, i+7):
					new_batch = [x for x in itertools.chain(new_batch, xy[j, 0:8])]
					if j == i+6:
						last = xy[j, [-1]]
				batch.append(new_batch)
				labels = np.append(labels, last)
				if i+7 == 7: # For testing purposes
					break
			else:
				continue			# If window size is not 7 or if the contents of the window is not continuous, skip

		batch = np.array(batch)

		self.x_data = batch
		self.y_data = labels
		'''
		print self.x_data
		print self.y_data
		print self.x_data.shape
		print self.y_data.shape
		print "\n\n"
		'''

	def __getitem__(self, index):
		window = torch.tensor(torch.from_numpy(self.x_data[index]))
		label = torch.tensor(self.y_data[index], dtype=torch.float64)
		#print window.type()
		#print label
		return window, label

	def __len__(self):
		return self.len


dataset = DiabetesDataset('./data/part1/Subject_2_part1.csv', './data/part1/list2_part1.csv')
train_loader = DataLoader(dataset=dataset,
						  batch_size=1,
						  shuffle=False,
						  num_workers=1)
'''
for epoch in range(2):
	for i, data in enumerate(train_loader, 0):
		# get the inputs
		inputs, labels = data

		# wrap them in Variable
		inputs, labels = Variable(inputs), Variable(labels)

		# Run your training process
print(epoch, i, "inputs", inputs.data, "labels", labels.data)
'''
# --- Global Statements End---

# https://pytorch.org/docs/master/nn.html
class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(1, 7*8*1)	
		self.fc1Drop = nn.Dropout(DROPOUT)	
		self.fc2 = nn.Linear(7*8*1, 7*8*1)
	
	def relu(self, x):
		#x = x.view(-1, 7*8*1)
		x = F.relu(self.fc1(x))
		x = self.fc1Drop(x)
		return F.log_softmax(self.fc2(x))

# --- Relu Functions Start ---

def train_relu(model, optimizer, epoch, log_interval = 100):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		if cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model.relu(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('RELU -- Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data.item()))
	
def validate_relu(model, optimizer, loss_vector, accuracy_vector):
	model.eval()
	val_loss, correct = 0, 0
	for data, target in validation_loader:
		output = model.relu(data)
		val_loss += F.nll_loss(output, target).data.item()
		pred = output.data.max(1)[1]
		correct += pred.eq(target.data).cpu().sum()

	val_loss /= len(validation_loader)
	loss_vector.append(val_loss)

	accuracy = 100 * correct / len(validation_loader.dataset)
	accuracy_vector.append(accuracy)
	print('\n\tValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		val_loss, correct, len(validation_loader.dataset), accuracy))

# --- Relu Functions End ---

def main():
	for rate in learningRates:
		model = Network()
		if cuda:
			model.cuda()
			
		print("Learning Rate: " + str(rate))

		optimizer = optim.SGD(model.parameters(), lr=rate)

		lossv, accv = [], []
		for epoch in range(1, epochs + 1):
			train_relu(model, optimizer, epoch)
			# validate_relu(model, optimizer, lossv, accv)

main()
