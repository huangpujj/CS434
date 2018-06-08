# Using Lam's code as a base to read and unpack data

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
	def get_indice(self, indice):
		self.indice = []
		for line in open(indice):
			self.indice.append(int(line))
	
	# Checks if window[start:end] is a continuous block
	def check_window(self, start, end):
		array = self.indice[start:end]
		for i, x in enumerate(array):
			if i + 1 < len(array):
				temp = x + 1
				if temp == array[i+1]:
					continue
				else:
					return False	# Window is not continuous
		return True					# Window is continuous

	def __init__(self, data, indice):
		xy = np.loadtxt(data, delimiter=',', usecols=(1, 2, 3, 4, 5, 6, 7, 8, 9), dtype=np.float32)
		self.len = xy.shape[0]
		batch = torch.tensor((), dtype=torch.float64)
		diag = torch.tensor((), dtype=torch.float64)
		
		self.get_indice(indice)

		concat_batch = []
		concat_diag = []

		for i, row in enumerate(xy):
			new_batch = []
			if  i+7 <= self.len and self.check_window(i, i+7):
				for j in range(i, i+7):
					new_batch = [x for x in itertools.chain(new_batch, xy[j, 0:8])]
					if j == i+6:
						last = xy[j, [-1]]
			else:
				continue
			
			concat_batch.append(new_batch)
			concat_diag = np.append(concat_diag, last)
		
		concat_batch = np.array(concat_batch)

		batch = torch.tensor(torch.from_numpy(concat_batch), dtype=torch.float64)
		diag = torch.tensor(torch.from_numpy(concat_diag), dtype=torch.float64)
		self.x_data = batch.float()
		self.y_data = diag

		print self.x_data
		print self.y_data
		print self.x_data.shape
		print self.y_data.shape
		print "\n\n"

	def __getitem__(self, index):
		return self.x_data[index], self.y_data[index]

	def __len__(self):
		return self.len


dataset = DiabetesDataset('./data/part1/Subject_2_part1.csv', './data/part1/list2_part1.csv')
train_loader = DataLoader(dataset=dataset,
						  batch_size=1,
						  shuffle=False,
						  num_workers=2)