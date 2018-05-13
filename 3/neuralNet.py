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

# Might not want to use nn?
class Network(nn.Module):
	def __init__(self):
	
	def forward(self x):
	
	# PyTorch will generate backward() automatically
	
model = Network()

optimizer = None	# Fill this in with a method from torch.optim

# train on data
def train(epoch, logInterval = None):
	model.eval()
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
	