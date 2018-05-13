# Using https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb 
# as a template for implementing PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as opt
from torchvision import datasets, transforms
from torch.autograde import Variable

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Not going to implement CUDA because why would we
batchSize = 32	# Might need to be changed later

# These loaders provide iterators over the datasets
trainLoader = torch.utils.data.DataLoader()	# fill in params
validationLoader = torch.utils.data.DataLoader() #fill in params

# https://pytorch.org/docs/master/nn.html
class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(32*32, 100)	# Hidden layer
		self.fc1Drop = nn.Dropout(0.2)		# Some regularization
		self.fc2 = nn.Linear(100, 10)		# Output layer, change second param
		
	def forward(self, x):
		x = x.view(-1, 32*32)				# is this necessary?
		x = F.sigmoid(self.fc1(x))
		x = self.fc1Drop(x)
		return F.log_softmax(self.fc2(x))
	
	# PyTorch will generate backward() automatically
	
model = Network()

learnRate = 0.01	# Change this value to change learning rates

optimizer = opt.SGD(model.parameters(), lr = learnRate, momentum = 0.5)	# what does momentum do?

# train on data
def train(epoch, logInterval = None):
	model.train()
	# code for train method goes here
	
# validate
def validate(lossVec, accVec):
	model.eval()
	# code for vlaidate method goes here
	
# Now we train and validate
epochs = None
lossVec, accVec = [], []
for epoch in range(1, epochs+1):
	train(epoch)
	validate(lossVec,accVec)
	
# Dating plotting can go below this
	