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
        features.append(line.split(",")[1:-1])  # Features in this array
    f.close()
    (diagnosis, features) = (np.array(diagnosis, dtype=int), np.array(features, dtype=float))
    return (diagnosis, features)

# Euclidian distance
def distance(x, xi):                 
    return np.linalg.norm(x-xi)

# Feature scaling normalization https://en.wikipedia.org/wiki/Feature_scaling
def normalize(x):
    x_max = np.amax(x, axis=0)
    x_min = np.amin(x, axis=0)     
    return (x - x_min) / (x_max - x_min)

# KNN 
def knn(train_f, test_f):
    dist = []
    for i in range(0, len(train_f)):                                    # For each training feature row, calculate its distance 
        dist.append( (train_f[i], distance(test_f, train_f[i])) )
    return sorted(dist,key=lambda x:(-x[1],x[0]))                       # Sort by distance in descending order

(train_d, train_f) = get_data('data/knn_train.csv')     # Get training set data
(test_d, test_f) = get_data('data/knn_test.csv')        # Get testing set data

train_f = normalize(train_f)
test_f = normalize(test_f)
print knn(train_f, test_f)
'''

_knn = []
for row in enumerate(test_f[0:-1]):
    _knn.append(knn(train_f, row[1]))
'''
#print _knn