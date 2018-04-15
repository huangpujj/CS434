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
    return (np.transpose(features).dot(outputs)).dot(np.linalg.inv(np.transpose(features).dot(features)))   # W = (X^T * X)^-1 * X^T * Y   

def ase(features, outputs, weight):                     # E(w) = (y - Xw)^T (y - Xw)
    sse = (outputs - np.dot(features, weight)).dot(np.transpose(outputs - np.dot(features, weight)))
    return sse/len(features)                            # ASE = SSE/N, where N = number of rows in dataset

def add_features(f, value = ""):
    if value == "":
        array_with_value = np.transpose(np.random.randint(1, 100, size=(1, len(f))))
    else:
        array_with_value = np.transpose(np.full((1, len(f)), value, dtype=float))
    return np.hstack((array_with_value, f))

def part1():
    seed = 69                                               # Random seed for replication
    np.set_printoptions(suppress=True)                      # Turn off scientific notation for CSV
    np.random.seed(seed)                                    

    (f_train, o_train) = get_data('data/housing_train.txt') # Read training data
    (f_test, o_test) = get_data('data/housing_test.txt')    # Read testing data

    w_train = weight(f_train, o_train)                      # Get training data weight
    w_test = weight(f_test, o_test)                         # Get testing data weight

    f_dummy_train = add_features(f_train, value = 1)        # Create dummy column for training data
    f_dummy_test = add_features(f_test, value = 1)          # Create dummy column for testing data
    w_train_dummy = weight(f_dummy_train, o_train)          # Get weight of dummy

    ase_train = ase(f_dummy_train, o_train, w_train_dummy)  # Get ASE of training data with dummy and dummy weight
    ase_test = ase(f_dummy_test, o_test, w_train_dummy)     # Get ASE of testing data with dummy and dummy weight

    print ("\n=====================================\n")
    print ("Weight Vector with Dummy Column\n" + str(w_train_dummy))    # Part 1.1.1
    print ("\nTraining ASE:\t" + str(ase_train))                        # Part 1.1.2
    print ("Testing ASE:\t" + str(ase_test))                            # Part 1.1.2

    ase_train = ase(f_train, o_train, w_train)          # Get ASE of training data with training data weight (no dummy)
    ase_test = ase(f_test, o_test, w_train)             # Get ASE of testing data with training data weight  (no dummy)

    print ("\n=====================================\n")
    print ("Weight Vector without Dummy Column\n" + str(w_train))   # Part 1.3.1 Remove dummy column
    print ("\nTraining ASE:\t" + str(ase_train))                    # Part 1.3.2
    print ("Testing ASE:\t" + str(ase_test))                        # Part 1.3.2

    # Part 1.4
    # Iterate from adding 2 to 100 random features and print ASE for training and testing
    print ("\n=====================================\n")
    print ("Random seed: " + str(seed))
    print("\td\tTraining ASE\tTesting ASE")
    f = open("ase.csv", 'w+')
    f.write("d,Training ASE,Testing ASE\n")
    for d in range(2, 100, 2):
        n_train_feature = f_train
        n_test_feature = f_test
        for j in range(1, d):
            n_train_feature = add_features(n_train_feature)
            n_test_feature = add_features(n_test_feature)
        n_train_weight = weight(n_train_feature, o_train)
        n_ase_train = ase(n_train_feature, o_train, n_train_weight)
        n_ase_test = ase(n_test_feature, o_test, n_train_weight)
        print("\t" + str(d) + "\t" + str(n_ase_train) + "\t" + str(n_ase_test) + "\t" + str(n_test_feature.shape))
        f.write(str(d) + "," + str(n_ase_train) + "," + str(n_ase_test) + "\n")
    f.close()
    print ("\n=====================================\n")

# Part 2.1 Gradient Descent
def get_data_csv(filename):
    rgb = []
    number = []
    f = open(filename)
    for line in f:
        rgb.append(line.split(",")[0:-1])                                           # features in rgb values
        number.append([line.split(",")[-1].replace("\n","")])                       # number they represent
    (rgb, number) = (np.array(rgb, dtype=float), np.array(number, dtype=float))     # Ensure all values are floats
    return (rgb, number)

def sigmoid(w, f):
    return 1.0 / (1.0 + np.exp((-1.0 * np.transpose(w)).dot(f)))                    # 1 / (1 + e^(-w^T x))

def gradient(w, f, o):
    g = np.zeros(256, dtype=float)
    for i in range(f.shape[0]):
        y_hat = sigmoid(w, f[i])                # Iterate over all features in each row
        g = g + (float(o[i]) - y_hat) * f[i]    # Reversed on slides
    return g

def batch_gradient_descent(itr, learning_rate, f_train, o_train, f_test, o_test):
    f = open("gradient_descent.csv", 'w+')
    print("Iteration\tTraining Accuracy\tTest Accuracy")
    f.write("Iteration,Training Accuracy,Test Accuracy\n")

    w = np.zeros(256, dtype=float) # Initilize w = [0, ...0]
    
    for i in range(1, itr):
        g = gradient(w, f_train, o_train)
        w = w + (learning_rate * g)
        print(str(i) + "\t" + str(check(w, f_train, o_train)) + "\t" + str(check(w, f_test, o_test)))
        f.write(str(i) + "," + str(check(w, f_train, o_train)) + "," + str(check(w, f_test, o_test)) + "\n")
    
    f.close()

def check(w, f, expected):
    correct = 0
    for i in range(0, f.shape[0]):
        y_hat = sigmoid(w, f[i])
        if np.round(y_hat) == expected[i]:
            correct += 1
    return float(correct) / float(f.shape[0])

def part2():
    itr = 169               # Training iterations
    learning_rate = 1       # Learning Rate

    (f_train, o_train) = get_data_csv("data/usps-4-9-train.csv")
    (f_test, o_test) = get_data_csv("data/usps-4-9-test.csv")
    
    f_train = np.divide(f_train, 255)   # Divide by 255 to avoid overflow
    f_test = np.divide(f_test, 255)
    
    batch_gradient_descent(itr, learning_rate, f_train, o_train, f_test, o_test)

part1()

part2()

