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
def knn(feature, train_diag, train_feat, k):
    dist = []                                               # Array that holds all pairs (feature, distance)
    d_arr = distance(feature, train_feat)                   # Calculate the distance from specific `feature` 
    for i in range(len(train_feat)):                        # to all other features.
        dist.append( (train_diag[i], d_arr[i]) )            # Put all features and distances in pairs
    sorted_dist = sorted(dist,key=lambda x:(x[1],-x[0]))    # Sort all paris in asceending order by distance
    k_sorted = sorted_dist[:k]                              # Return the closest k pairs out
    return k_sorted

# Determines what diagnosis `feature` is based on its k closest neighbors
def classify(feature, train_diag, train_feat, k):
    _knn = knn(feature, train_diag, train_feat, k)      # Get all closest neighbors.
    total = 0
    for i in range(len(_knn)):                      # Sum all neighbors.
        total += _knn[i][0]                         # If majority of neighbors are
    if total < 0:                                   # negative then total will be less
        return -1                                   # than 0, so we assume `feature`
    else:                                           # is also negative. Otherwise assume
        return 1                                    # it's positive

def knn_error(diag_set, feat_set, train_diag, train_feat, k):
    predict = []
    for i in range(len(feat_set)):
        predict.append( classify(feat_set[i], train_diag, train_feat, k) )
    return (np.sum(np.abs(predict - diag_set)) / float(2 * len(diag_set))) * 100

# Leave-one-out cross-validation error
def L1O_cross_valid_error(diag_set, feat_set, k):
    predict = []
    for i in range(len(feat_set)):
        predict.append( classify(feat_set[i], np.delete(diag_set, i, axis=0), np.delete(feat_set, i, axis=0), k) )
    return (np.sum(np.abs(predict - diag_set)) / float(2 * len(diag_set))) * 100


# --- Main ---
train_d, train_f = get_data('data/knn_train.csv')     # Get training set data
test_d, test_f = get_data('data/knn_test.csv')        # Get testing set data

train_f = normalize(train_f)                          # Normalize using feature scaling
test_f = normalize(test_f)

f = open("part1.csv", 'w+')

f.write("k,Training Error,Testing Error,Leave-one-out Cross-validation Error\n")
print("k\tTraining Error\tTesting Error\tLeave-one-out Cross-validation Error")

for k in range(1, 80, 2): # k values 1, 3, 5 ... 51
    train_error = str( "{0:.3f}".format(round(knn_error(train_d, train_f, train_d, train_f, k),3)) ) + "%"
    test_error  = str( "{0:.3f}".format(round(knn_error(test_d, test_f, train_d, train_f, k),3)) )   + "%"
    L1O_error   = str( "{0:.3f}".format(round(L1O_cross_valid_error(train_d, train_f, k),3)) ) + "%"
    
    print( str(k) + "\t" + train_error + "\t\t" + test_error + "\t\t" + L1O_error )
    f.write( str(k) + "," + train_error + "," + test_error + "," + L1O_error + "\n")

f.close()