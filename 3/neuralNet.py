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
		self.fcm1 = nn.Linear(3*32*32, 50)
		self.fcm2 = nn.Linear(50,50)
		self.fcm3 = nn.Linear(50,10)
		
	def sigmoid(self, x):
		x = x.view(-1, 3*32*32)
		x = F.sigmoid(self.fc1(x))
		x = self.fc1Drop(x)
		x = F.sigmoid(self.fc2(x))
		return x
	
	def relu(self,x):
		x = x.view(-1, 3*32*32)
		x = F.relu(self.fc1(x))
		x = self.fc1Drop(x)
		x = F.relu(self.fc2(x))
		return x

	def multi_layer(self,x):
		x = x.view(-1, 3*32*32)
		x = F.relu(self.fcm1(x))
		x = self.fc1Drop(x)
		x = F.relu(self.fcm2(x))
		x = self.fc1Drop(x)
		x = F.relu(self.fcm3(x))
		return x


# --- Sigmoid Functions Start ---
def train_sigmoid(model, optimizer, epoch, log_interval = 100):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		if cuda:
			data, target = data.cuda(), target.cuda()
		data, target = Variable(data), Variable(target)
		optimizer.zero_grad()
		output = model.sigmoid(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data.item()))

def validate_sigmoid(model, optimizer, sig_accuracy, sig_avg_loss,loss_vector, accuracy_vector):
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
	sig_avg_loss.write("{:.4f},".format(val_loss))
	sig_accuracy.write("{:.0f}%,".format(accuracy))

# --- Sigmoid Functions End ---

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
	
def validate_relu(model, optimizer, relu_accuracy, relu_avg_loss, loss_vector, accuracy_vector):
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
	
	relu_avg_loss.write("{:.4f},".format(val_loss))
	relu_accuracy.write("{:.0f}%,".format(accuracy))

# --- Relu Functions End ---

# --- Multi Functions Start ---

def train_multi(model, optimizer, epoch, log_interval = 100):
	model.train()
	for batch_idx, data in enumerate(train_loader, 0):
		inputs, labels = data
		#if cuda:
		#	data, target = data.cuda(), target.cuda()
		optimizer.zero_grad()
		output = model.multi_layer(inputs)
		loss = criterion(output, labels)
		loss.backward()
		optimizer.step()
		if batch_idx % log_interval == 0:
			print('\t Multi_Layer -- Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
				epoch, batch_idx * len(data), len(train_loader.dataset),
				100 * batch_idx / len(train_loader), loss.item()))
	
def validate_multi(model, optimizer, relu_accuracy, relu_avg_loss, loss_vector, accuracy_vector):
	model.eval()
	val_loss, correct = 0, 0
	for data, target in validation_loader:
		output = model.multi_layer(data)
		val_loss += F.nll_loss(output, target).data.item()
		pred = output.data.max(1)[1]
		correct += pred.eq(target.data).cpu().sum()

	val_loss /= len(validation_loader)
	loss_vector.append(val_loss)

	accuracy = 100 * correct / len(validation_loader.dataset)
	accuracy_vector.append(accuracy)
	print('\n\tValidation set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		val_loss, correct, len(validation_loader.dataset), accuracy))
	
	relu_avg_loss.write("{:.4f}%,".format(val_loss))
	relu_accuracy.write("{:.0f}%,".format(accuracy))

# --- Multi Functions End ---

def part1():
	p1_sig_acc = open("p1_sig_accuracy.csv", 'w+')
	p1_sig_acc.write("Sigmoid Accuracy\n")
	p1_sig_acc.write("Epochs, ")

	p1_sig_avg_loss = open("p1_sig_avg_loss.csv", 'w+')
	p1_sig_avg_loss.write("Sigmoid Average Loss\n")
	p1_sig_avg_loss.write("Epochs, ")

	for i in range(1, epochs + 1):
		p1_sig_acc.write(str(i) + ",")
		p1_sig_avg_loss.write(str(i) + ",")
		
	p1_sig_acc.write("\n")
	p1_sig_avg_loss.write("\n")

	for rate in learningRates:
		model = Network()
		if cuda:
			model.cuda()
			
		print("Learning Rate: " + str(rate))
		p1_sig_acc.write("LR: " + str(rate) + ",")
		p1_sig_avg_loss.write("LR: " + str(rate) + ",")

		optimizer = optim.SGD(model.parameters(), lr=rate)

		lossv, accv = [], []
		for epoch in range(1, epochs + 1):
			train_sigmoid(model, optimizer, epoch)
			validate_sigmoid(model, optimizer, p1_sig_acc, p1_sig_avg_loss, lossv, accv)
		p1_sig_acc.write("\n")
		p1_sig_avg_loss.write("\n")

	p1_sig_acc.close()
	p1_sig_avg_loss.close()

def part2():
	p2_relu_acc = open("p2_relu_accuracy.csv", 'w+')
	p2_relu_acc.write("Relu Accuracy\n")
	p2_relu_acc.write("Epochs, ")

	p2_relu_avg_loss = open("p2_relu_avg_loss.csv", 'w+')
	p2_relu_avg_loss.write("Relu Average Loss\n")
	p2_relu_avg_loss.write("Epochs, ")

	for i in range(1, epochs + 1):
		p2_relu_acc.write(str(i) + ",")
		p2_relu_avg_loss.write(str(i) + ",")
		
	p2_relu_acc.write("\n")
	p2_relu_avg_loss.write("\n")

	for rate in learningRates:
		model = Network()
		if cuda:
			model.cuda()
			
		print("Learning Rate: " + str(rate))
		p2_relu_acc.write("LR: " + str(rate) + ",")
		p2_relu_avg_loss.write("LR: " + str(rate) + ",")

		optimizer = optim.SGD(model.parameters(), lr=rate)

		lossv, accv = [], []
		for epoch in range(1, epochs + 1):
			train_relu(model, optimizer, epoch)
			validate_relu(model, optimizer, p2_relu_acc, p2_relu_avg_loss, lossv, accv)
		p2_relu_acc.write("\n")
		p2_relu_avg_loss.write("\n")

	p2_relu_acc.close()
	p2_relu_avg_loss.close()

def part3():
	part3 = open("part3_relu_improved.csv", 'w+')
	part3.write("Relu\n")
	part3.write("Epochs, ")
	for i in range(1, epochs + 1):
		part3.write(str(i) + ",")
	part3.write("\n")

	for rate in learningRates:
		model = Network()
		if cuda:
			model.cuda()
			
		print("Learning Rate: " + str(rate))
		part3.write(str(rate) + ",")
		optimizer = optim.SGD(model.parameters(), lr=rate, momentum=0, weight_decay=0.5)

		lossv, accv = [], []
		for epoch in range(1, epochs + 1):
			train_relu(model, optimizer, epoch)
			validate_relu(model, optimizer, part3, lossv, accv)
		part3.write("\n")
	part3.close()

def part4():
	part4 = open("part4_multi_layer.csv", 'w+')
	part4.write("Relu\n")
	part4.write("Epochs, ")
	for i in range(1, epochs + 1):
		part4.write(str(i) + ",")
	part4.write("\n")

	for rate in learningRates:
		model = Network()
		if cuda:
			model.cuda()
			
		print("Learning Rate: " + str(rate))
		part4.write(str(rate) + ",")
		optimizer = optim.SGD(model.parameters(), lr=rate, momentum=0.5, weight_decay=0.5)

		lossv, accv = [], []
		for epoch in range(1, epochs + 1):
			train_multi(model, optimizer, epoch)
			validate_multi(model, optimizer, part4, lossv, accv)
		part4.write("\n")
	part4.close()
'''
print("\t--- Part 1 Start ---\n")
part1()
print("\t--- Part 1 End ---\n")
'''
print("\t--- Part 2 Start ---\n")
part2()
print("\t--- Part 2 End ---\n")
'''
print("\t--- Part 3 Start ---\n")
part3()
print("\t--- Part 3 End ---\n")

print("\t--- Part 4 Start ---\n")
part4()
print("\t--- Part 4 End ---\n")
'''