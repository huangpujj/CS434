import numpy as np
import random

def get_data(filename):
    features = []
    outputs = []
    f = open(filename, 'r')
    for line in f:
        features.append(line.split(",")[0])
        outputs.append(line.split(",")[1:-1])
    f.close()
    (features, outputs) = (np.array(features, dtype=float), np.array(outputs, dtype=float))
    return (features, outputs)

print get_data('data/knn_train.csv')