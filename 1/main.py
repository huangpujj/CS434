# Kin-Ho Lam (ONID\lamki) | CS 434 | 4/10/18

import numpy as np
import random
import StringIO

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

def weight(features, outputs):
    f_tf = np.matmul(np.transpose(features), features)  # X^T * X
    inverse = np.linalg.inv(f_tf)                       # (X^T * X)^-1
    f_to = np.matmul(np.transpose(features), outputs)   # (X^T * X)^-1 * X^T 
    return np.matmul(inverse, f_to)                     # W = (X^T * X)^-1 * X^T * Y   

def sse(features, outputs, weight):                     # E(w) = (y - Xw)^T (y - Xw)
    return np.matmul(np.transpose(outputs - np.matmul(features, weight)), outputs - np.matmul(features, weight))

def add_features(f, value = ""):
    if value == "":
        array_with_value = np.transpose(np.random.randint(1, 100, size=(1, len(f))))
    else:
        array_with_value = np.transpose(np.full((1, len(f)), value, dtype=float))
    return np.hstack((array_with_value, f))

# main
np.set_printoptions(suppress=True)

np.random.seed(1)

(f_train, o_train) = get_data('data/housing_train.txt') # Read training data
(f_test, o_test) = get_data('data/housing_test.txt')    # Read testing data

w_train = weight(f_train, o_train)                      # Get training data weight
w_test = weight(f_test, o_test)                         # Get testing data weight

f_dummy_train = add_features(f_train, 1)                # Create dummy column for training data
w_train_dummy = weight(f_dummy_train, o_train)          # Get weight of dummy

f_dummy_test = add_features(f_test, 1)                  # Create dummy column for testing data

sse_train = sse(f_dummy_train, o_train, w_train_dummy)  # Get SSE of training data with dummy and dummy weight
sse_test = sse(f_dummy_test, o_test, w_train_dummy)     # Get SSE of testing data with dummy and dummy weight

print ("\n=====================================\n")
print ("Weight Vector with Dummy Column\n" + str(w_train_dummy))    # Part 1.1
print ("\nTraining SSE:\t" + str(sse_train))                        # Part 1.2
print ("Testing SSE:\t" + str(sse_test))                            # Part 1.2

sse_train = sse(f_train, o_train, w_train)          # Get SSE of training data with training data weight (no dummy)
sse_test = sse(f_test, o_test, w_train)             # Get SSE of testing data with training data weight  (no dummy)

print ("\n=====================================\n")
print ("Weight Vector without Dummy Column\n" + str(w_train))   # Part 3.1 Remove dummy column
print ("\nTraining SSE:\t" + str(sse_train))                    # Part 3.2
print ("Testing SSE:\t" + str(sse_test))                        # Part 3.2


# Part 4
# Iterate from adding 2 to 100 random features and print SSE for training and testing
print ("\n=====================================\n")
print("\td\tSSE Training\tSSE Test")
f = open("SSE.csv", 'w+')
f.write("d,SSE Training,SSE Test\n")
for d in range(2, 60):
    n_train_feature = f_train
    n_test_feature = f_test
    for j in range(1, d):
        n_train_feature = add_features(n_train_feature)
        n_test_feature = add_features(n_test_feature)
    n_train_weight = weight(n_train_feature, o_train)
    n_test_weight = weight(n_test_feature, o_test)
    n_sse_train = sse(n_train_feature, o_train, n_train_weight)
    n_sse_test = sse(n_test_feature, o_test, n_test_weight)
    print("\t" + str(d) + "\t" + str(n_sse_train) + "\t" + str(n_sse_test))
    f.write(str(d) + "," + str(n_sse_train) + "," + str(n_sse_test) + "\n")
f.close()
print ("\n=====================================\n")