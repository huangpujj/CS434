# Using https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb 
# as a template for implementing PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable	# Deprecated

import numpy as np
#import matplotlib.pyplot as plt

# https://pytorch.org/docs/master/nn.html
class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(3*32*32, 100)	# Hidden layer
		self.fc1Drop = nn.Dropout(0.2)		# Some regularization
		self.fc2 = nn.Linear(100, 10)
		#self.fc3 = nn.Linear(50, 10)
		
	def forward(self, x):
		x = x.view(-1, 3*32*32)
		x = F.sigmoid(self.fc1(x))
		x = self.fc1Drop(x)
		x = F.sigmoid(self.fc2(x))
		#x = F.sigmoid(self.fc3(x))
		return x
	
learningRates = [0.0001, 0.001, 0.01, 0.1]
epochs = 10
batch_size = 32

criterion = nn.CrossEntropyLoss()
cuda = torch.cuda.is_available()

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

train_loader = torch.utils.data.DataLoader(
	datasets.CIFAR10('../data', train=True, download=True,
				   transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize([0.53129727, 0.5259391, 0.52069134], [0.28938246, 0.28505746, 0.27971658])
				   ])),
	batch_size=batch_size, shuffle=True, num_workers=2)

validation_loader = torch.utils.data.DataLoader(
	datasets.CIFAR10('../data', train=False, download=True, transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize([0.53129727, 0.5259391, 0.52069134], [0.28938246, 0.28505746, 0.27971658])
				   ])),
	batch_size=batch_size, shuffle=False, num_workers=2)

# train on data
def train(epoch, log_interval = 100):
	model.train()
	for batch_idx, data in enumerate(train_loader, 0):
		inputs, labels = data
		#if cuda:
		#	data, target = data.cuda(), target.cuda()
		optimizer.zero_grad()
		output = model(inputs)
		loss = criterion(output, labels)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.item()))
	
# validate
def validate(loss_vector, accuracy_vector):
	model.eval()
	val_loss, correct = 0, 0
	for data, target in validation_loader:
		output = model(data)
		val_loss += F.nll_loss(output, target).data.item()
		pred = output.data.max(1)[1] 					# get the index of the max log-probability
		correct += pred.eq(target.data).cpu().sum()

	val_loss /= len(validation_loader)
	loss_vector.append(val_loss)

	accuracy = 100. * correct / len(validation_loader.dataset)
	accuracy_vector.append(accuracy)
	
	print('\n\tValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		val_loss, correct, len(validation_loader.dataset), accuracy))

model = Network()
if cuda:
	model.cuda()

for rate in learningRates:
	print("Learning Rate: " + str(rate))
	optimizer = optim.SGD(model.parameters(), lr=rate, momentum=0.5)

	lossv, accv = [], []
	for epoch in range(1, epochs + 1):
		train(epoch)
		validate(lossv, accv)
	
# Dating plotting can go below this