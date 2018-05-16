# Using https://github.com/CSCfi/machine-learning-scripts/blob/master/notebooks/pytorch-mnist-mlp.ipynb 
# as a template for implementing PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable	# Deprecated

import numpy as np

# --- Global Statements Start ---

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

# --- Global Statements End---

# https://pytorch.org/docs/master/nn.html
class Network(nn.Module):
	def __init__(self):
		super(Network, self).__init__()
		self.fc1 = nn.Linear(3*32*32, 100)	# Hidden layer
		self.fc1Drop = nn.Dropout(0.2)		# Some regularization
		self.fc2 = nn.Linear(100, 10)
		#self.fc3 = nn.Linear(50, 10)
		
	def sigmoid(self, x):
		x = x.view(-1, 3*32*32)
		x = F.sigmoid(self.fc1(x))
		x = self.fc1Drop(x)
		x = F.sigmoid(self.fc2(x))
		#x = F.sigmoid(self.fc3(x))
		return x
	
	def relu(self,x):
		x = x.view(-1, 3*32*32)
		x = F.relu(self.fc1(x))
		x = self.fc1Drop(x)
		x = F.relu(self.fc2(x))
		return x

# --- Sigmoid Functions Start ---
def train_sigmoid(model, optimizer, epoch, log_interval = 100):
	model.train()
	for batch_idx, data in enumerate(train_loader, 0):
		inputs, labels = data
		optimizer.zero_grad()
		output = model.sigmoid(inputs)
		loss = criterion(output, labels)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('\tTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100 * batch_idx / len(train_loader), loss.item()))

def validate_sigmoid(model, optimizer, filename, loss_vector, accuracy_vector):
	model.eval()
	val_loss, correct = 0, 0
	for data, target in validation_loader:
		output = model.sigmoid(data)
		val_loss += F.nll_loss(output, target).data.item()
		pred = output.data.max(1)[1]
		correct += pred.eq(target.data).cpu().sum()

	val_loss /= len(validation_loader)
	loss_vector.append(val_loss)

	accuracy = 100 * correct / len(validation_loader.dataset)
	accuracy_vector.append(accuracy)
	
	print('\n\tValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		val_loss, correct, len(validation_loader.dataset), accuracy))
	filename.write("{:.0f}%,".format(accuracy))

# --- Sigmoid Functions End ---

# --- Relu Functions Start ---

def train_relu(model, optimizer, epoch, log_interval = 100):
	model.train()
	for batch_idx, data in enumerate(train_loader, 0):
		inputs, labels = data
		#if cuda:
		#	data, target = data.cuda(), target.cuda()
		optimizer.zero_grad()
		output = model.relu(inputs)
		loss = criterion(output, labels)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('\t RELU -- Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100 * batch_idx / len(train_loader), loss.item()))
	
def validate_relu(model, optimizer, filename, loss_vector, accuracy_vector):
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
	filename.write("{:.0f}%,".format(accuracy))

# --- Relu Functions End ---

def part1():
	part1 = open("part1_sigmoid.csv", 'w+')
	part1.write("Sigmoid\n")
	part1.write("Epochs, ")
	for i in range(1, epochs + 1):
		part1.write(str(i) + ",")
	part1.write("\n")

	for rate in learningRates:
		model = Network()
		if cuda:
			model.cuda()
			
		print("Learning Rate: " + str(rate))
		part1.write(str(rate) + ",")
		optimizer = optim.SGD(model.parameters(), lr=rate, momentum=0.5)

		lossv, accv = [], []
		for epoch in range(1, epochs + 1):
			train_sigmoid(model, optimizer, epoch)
			validate_sigmoid(model, optimizer, part1, lossv, accv)
		part1.write("\n")
	part1.close()

def part2():
	part2 = open("part2_relu.csv", 'w+')
	part2.write("Relu\n")
	part2.write("Epochs, ")
	for i in range(1, epochs + 1):
		part2.write(str(i) + ",")
	part2.write("\n")

	for rate in learningRates:
		model = Network()
		if cuda:
			model.cuda()
			
		print("Learning Rate: " + str(rate))
		part2.write(str(rate) + ",")
		optimizer = optim.SGD(model.parameters(), lr=rate, momentum=0.5)

		lossv, accv = [], []
		for epoch in range(1, epochs + 1):
			train_relu(model, optimizer, epoch)
			validate_relu(model, optimizer, part2, lossv, accv)
		part2.write("\n")
	part2.close()

print("\t--- Part 1 Start ---\n")
part1()
print("\t--- Part 1 End ---\n")

print("\t--- Part 2 Start ---\n")
part2()
print("\t--- Part 2 End ---\n")