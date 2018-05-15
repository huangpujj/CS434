# Using https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb 
# as a template for implementing PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torchvision import datasets, transforms
#from torch.autograde import Variable	# Depreciated

import numpy as np
#import matplotlib.pyplot as plt
import seaborn as sns

learnRate = [0.01]	# Change this value to change learning rates

cuda = torch.cuda.is_available()
batch_size = 32	# Might need to be changed later

kwargs = {'num_workers': 1, 'pin_memory': True} if cuda else {}

# These loaders provide iterators over the datasets
train_loader = torch.utils.data.DataLoader(
	datasets.MNIST('cifar', train=True, download=True,
				   transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize([0.53129727, 0.5259391, 0.52069134], [0.28938246, 0.28505746, 0.27971658])
				   ])),
	batch_size=batch_size, shuffle=True, **kwargs)


validation_loader = torch.utils.data.DataLoader(
	datasets.MNIST('cifar', train=False, transform=transforms.Compose([
					   transforms.ToTensor(),
					   transforms.Normalize([0.53129727, 0.5259391, 0.52069134], [0.28938246, 0.28505746, 0.27971658])
				   ])),
	batch_size=batch_size, shuffle=False, **kwargs) #fill in params

# https://pytorch.org/docs/master/nn.html
class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(32*32, 100)	# Hidden layer
		self.fc1Drop = nn.Dropout(0.2)		# Some regularization
		self.fc2 = nn.Linear(100, 10)		# Output layer, change second param
		
	def forward(self, x):
		x = F.sigmoid(self.fc1(x))
		x = self.fc1Drop(x)
		return F.log_softmax(self.fc2(x))
	
	# PyTorch will generate backward() automatically

model = Network()
if cuda:
	model.cuda()

# train on data
def train(epoch, logInterval = None):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		#data, target = Variable(data), Variable(target)	# Depreciated
		if cuda:
			data, target = data.cuda(), target.cuda()
		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100. * batch_idx / len(train_loader), loss.data[0]))
	
# validate
def validate(lossVec, accVec):
	model.eval()
	val_loss, correct = 0, 0
	for data, target in validation_loader:
		#data, target = Variable(data, volatile=True), Variable(target)	 # Depreciated
		if cuda:
			data, target = data.cuda(), target.cuda()
		output = model(data)
		val_loss += F.nll_loss(output, target).data[0]
		pred = output.data.max(1)[1] # get the index of the max log-probability
		correct += pred.eq(target.data).cpu().sum()

	val_loss /= len(validation_loader)
	loss_vector.append(val_loss)

	accuracy = 100. * correct / len(validation_loader.dataset)
	accuracy_vector.append(accuracy)
	
	print('\n\tValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		val_loss, correct, len(validation_loader.dataset), accuracy))

for rate in learnRate:
	print("\nLearning Rate: " + str(rate))
	epochs = 10
	optimizer = opt.SGD(model.parameters(), lr = rate, momentum = 0.5)	# what does momentum do?
	lossVec, accVec = [], []
	for epoch in range(1, epochs+1):
		train(epoch)
		validate(lossVec,accVec)
	
# Dating plotting can go below this
	