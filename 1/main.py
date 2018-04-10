import numpy as np
import random
import StringIO

def get_data(filename):
    f = open(filename, 'r')
    features = []
    outputs = []
    for line in f:
        features.append(line.split()[0:-1]) # Column 1-13 are features (crime rate, accessibility etc)
        outputs.append(line.split()[-1])    # Column 14 is the goal (median housing value)
    f.close()
    (features, outputs) = (np.array(features, dtype=float), np.array(outputs, dtype=float)) # Ensure all values are floats
    return (features, outputs)

def get_weight(features, outputs):
    f_transpose_f = np.matmul(np.transpose(features), features)
    inverse = np.linalg.inv(f_transpose_f)
    f_transpose_o = np.matmul(np.transpose(features), outputs)
    weight = np.matmul(inverse, f_transpose_o)
    return weight

# PART 1
def p1():
    (features, outputs) = get_data('data/housing_train.txt')
    weight = get_weight(features, outputs)
    print ("PART 1\n\tLearned Weight Vector:\n" + str(weight))

p1()