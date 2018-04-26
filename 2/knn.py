import numpy as np
import random

def get_data(filename):
    features = []
    outputs = []
    f = open(filename, 'r')
    for line in f:
        features.append(line.split()[0:-1]) # Column 1-13 are features (crime rate, accessibility etc)
        outputs.append(line.split()[-1])    # Column 14 is the goal (median housing value)
    f.close()
    (features, outputs) = (np.array(features, dtype=float), np.array(outputs, dtype=float)) # Ensure all values are floats
    return (features, outputs)

print get_data('data/knn_train.csv')