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

def get_sse(features, outputs, weight):                 # E(w) = (y - Xw)_t (y - Xw)
    sse = np.matmul(np.transpose(outputs - np.matmul(features, weight)), outputs - np.matmul(features, weight))
    return sse

# PART 1
def p1():
    (features, outputs) = get_data('data/housing_train.txt')    # Read training data
    weight = get_weight(features, outputs)                      # Get training data weight

    fill_ones = np.ones((1, len(features)), dtype=float)
    fill_ones_t = np.transpose(fill_ones)
    weight_with_dummy_vars = get_weight( np.hstack((fill_ones_t, features)), outputs)      
    
    print ("\nPART 1")
    print ("\n\tTraining Weight:\n" + str(weight))
    print ("\n\tTraining Weight with dummy variables:\n" + str(weight_with_dummy_vars))

# PART 2
def p2(weight):
    (f_train, o_train) = get_data('data/housing_train.txt') # Read training data
    (f_test, o_test) = get_data('data/housing_test.txt')    # Read testing data

    w_train = get_weight(f_train, o_train)                  # Get training data weight
    w_test = get_weight(f_test, o_test)                     # Get testing data weight

    sse_train = get_sse(f_train, o_train, w_train)
    sse_test = get_sse(f_test, o_test, w_test)
    print ("\nPART 2")
    print ("\tTesting Weight:\n" + str(w_test))
    print ("\n\tTraining SSE:\t" + str(sse_train))
    print ("\tTesting SSE:\t" + str(sse_test))

weight = p1()

p2(weight)
