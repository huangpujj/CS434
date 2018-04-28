import numpy as np
import operator
from operator import itemgetter

# Fetches data from filename and returns a tuple (diagnosis, features) where 
# diagnosis and features are arrays
def get_data(filename):
    diagnosis = []
    features = []
    f = open(filename, 'r')
    for line in f:
        diagnosis.append(line.split(",")[0])    # Diagnosis in this array
        features.append(line.split(",")[1:])  # Features in this array
    f.close()
    (diagnosis, features) = (np.array(diagnosis, dtype=int), np.array(features, dtype=float))
    return diagnosis, features

# Euclidian distance
# https://stackoverflow.com/questions/4370975/python-numpy-euclidean-distance-calculation-between-matrices-of-row-vectors
def distance(x, xi):
    return np.sqrt( np.sum((x - xi)**2, axis=1) )

# Feature scaling normalization https://en.wikipedia.org/wiki/Feature_scaling
def normalize(x):
    x_max = np.amax(x, axis=0)
    x_min = np.amin(x, axis=0)     
    return (x - x_min) / (x_max - x_min)

# KNN 
def knn(feature, diagnosis, features, k):
    dist = []
    d_arr = distance(feature, features)
    for i in range(len(features)):
        dist.append( (diagnosis[i], d_arr[i]) )
    sorted_dist = sorted(dist,key=lambda x:(x[1],-x[0]))
    return sorted_dist

def classify(feature, diagnosis, features, k):
    _knn = knn(feature, diagnosis, features, k)

def training_error(train_d, train_f, test_d, test_f, K):
    predict = []
    for i in range(len(train_f)):
        predict.append( classify(train_f[i], train_d, train_f, K) )

train_d, train_f = get_data('data/knn_train.csv')     # Get training set data
test_d, test_f = get_data('data/knn_test.csv')        # Get testing set data

train_f = normalize(train_f)
test_f = normalize(test_f)

K = range(1, 51, 2) # K values 1, 3, 5 ... 51

print("\tTraining Error:\t" + str(training_error(train_d, train_f, test_d, test_f, K) ) )