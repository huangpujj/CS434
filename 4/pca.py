import numpy as np
import matplotlib

# This backend allows saving figures without a display
matplotlib.use('Agg')
np.set_printoptions(threshold=np.inf)

import matplotlib.pyplot as plt

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
		center_val = row-center
		var = np.matmul(center_val,center_val.T)
		res = np.add(res,var)
	return res/n

def find_top(e_val, e_vect):
	n = 10
	idx = e_val.argsort()[::-1]
	sorted_eigenValues = e_val[idx]
	sorted_eigenVectors = e_vect[:,idx]
	tt_eigenVal = sorted_eigenValues[:n]
	tt_eigenVect = sorted_eigenVectors[:n]
	return tt_eigenVal, tt_eigenVect

def main():
	data = np.genfromtxt("data/data-1.txt", delimiter=',')
	
	#calculates center of data
	mu = calc_center(data)
	
	plt.plot(mu)
	plt.savefig('center.pdf')
	#print("Center:\n", mu)
	np.savetxt("center.csv", mu, delimiter=",")

	#calculates the covariance
	cov = calc_covariance(data,mu)
	#print("Covariance:\n", cov)
	np.savetxt("covariance.csv", cov, delimiter=",")

	#find eigenvalues/eigenvectors
	e_val, e_vect = np.linalg.eig(cov)
	top_val, top_vect = find_top(e_val,e_vect)
	#plt.plot(top_val)
	plt.clf()
	plt.plot(top_vect)
	plt.savefig('eigenvectors.pdf')
	#print("Top 10 Eigen Values:\n", top_val)
	#print("Top 10 Eigen Vectors:\n", top_vect)
	np.savetxt("eigenvalue.csv", top_val, delimiter=",")
	np.savetxt("eigenvector.csv", top_vect, delimiter=",")
	return

if __name__ == "__main__":
	main()