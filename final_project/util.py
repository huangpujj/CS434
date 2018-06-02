import numpy as np

def getData(fname):
	data = np.genfromtxt(fname, delimiter=',')
	return data