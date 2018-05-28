import numpy as np
import matplotlib

# This backend allows saving figures without a display
matplotlib.use('Agg')
np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt

MIN_INT = -21474836348
TOP = 10
DATA = 6000
DIMENSION = 784

def calc_center(data):
	res = data[0]
	n = len(data)
	for i in range(1,n):
		res += data[i]
	return res/n

def calc_covariance(data,center):
	res = np.zeros((DIMENSION, DIMENSION))
	n = len(data)
	for row in data:
		var = np.matmul(row,row.T)
		res = np.add(res,var)
	return res/n

def find_top(e_val, e_vect):
	idx = e_val.argsort()[::-1]
	sorted_eigenValues = e_val[idx]
	sorted_eigenVectors = e_vect[:,idx]
	tt_eigenVal = sorted_eigenValues[:TOP]
	tt_eigenVect = sorted_eigenVectors[:,:TOP]
	trans_tt_eigenVect = sorted_eigenVectors[:,:TOP].T
	return tt_eigenVal, tt_eigenVect, trans_tt_eigenVect

def find_largest_val(vect,center_values):
	val = np.full(TOP, MIN_INT, dtype=float)
	idx = np.full(TOP, 0, dtype=int)
	for i in range(DATA):
		temp = np.matmul(vect, center_values[i])
		for j in range(TOP):
			if temp[j] > val[j]:
				val[j] = temp[j]
				idx[j] = i
	return idx

def plot_largest_val(idx, center_values):
	for i in range(TOP):
		plt.clf()
		plt.imshow(np.reshape(center_values[idx[i]], (28, 28)))
		plt.savefig("pca_output/part3_3/Image" + str(i+1) + ".png")

def plot_eigenvect(vect):
	for i in range(TOP):
		plt.clf()
		img = np.reshape(vect[i], (28, 28))
		plt.imshow(img)
		plt.savefig("pca_output/part3_2/eigenVector" + str(i+1) + ".png")

def plot_mean(mu):
	plt.clf()
	plt.imshow(np.reshape(mu, (28, 28)))
	plt.savefig("pca_output/part3_2/mean.png")

def main():

	data = np.genfromtxt("data/data-1.txt", delimiter=',')

	#calculates center of data
	mu = calc_center(data)
	
	#print("Center:\n", mu)
	np.savetxt("pca_output/part3_1/center.csv", mu, delimiter=",")

	#calculates the covariance
	center_values = data - mu
	cov = np.cov(center_values, rowvar=False)
	#cov = calc_covariance(center_values,mu)
	#print("Covariance:\n", cov)
	np.savetxt("pca_output/part3_1/covariance.csv", cov, delimiter=",")

	#find eigenvalues/eigenvectors
	e_val, e_vect = np.linalg.eigh(cov)
	top_val, top_vect, trans_top_vect= find_top(e_val,e_vect)

	#graphs the mean
	plot_mean(mu)
	
	#graphs the eigenvectors
	plot_eigenvect(trans_top_vect)
	
	#find image with the largest value and plot it
	idx = find_largest_val(trans_top_vect,center_values)
	plot_largest_val(idx, center_values)


	#print("Top 10 Eigen Values:\n", top_val)
	#print("Top 10 Eigen Vectors:\n", top_vect)
	np.savetxt("pca_output/part3_1/eigenvalue.csv", top_val, delimiter=",")
	np.savetxt("pca_output/part3_1/eigenvector.csv", top_vect, delimiter=",")
	return

if __name__ == "__main__":
	main()