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
    f_tf = np.matmul(np.transpose(features), features)  # X_t * X
    inverse = np.linalg.inv(f_tf)                       # (X_t * X)^-1
    f_to = np.matmul(np.transpose(features), outputs)   # (X_t * X)^-1 * X_t 
    weight = np.matmul(inverse, f_to)                   # W = (X_t * X)^-1 * X_t * Y   
    return weight

def get_sse(weight):
    return weight

# PART 1
def p1():
    (features, outputs) = get_data('data/housing_train.txt')
    weight = get_weight(features, outputs)
    print ("\nPART 1\n\tLearned Weight Vector:\n" + str(weight))
    return weight

# PART 2
def p2(weight):
    sse = get_sse(weight)
    print ("\nPART 2\n\t")
    return weight

weight = p1()

p2(weight)
