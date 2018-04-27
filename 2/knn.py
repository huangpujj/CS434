import numpy as np

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

def dist(x, y):                 # Distance Function
    return np.linalg.norm(x-y)

(diagnosis, features) = get_data('data/knn_train.csv')

print("Diagnosis: " + str(diagnosis))   # Sanity check
print("Features: " + str(features) + "\n")

print dist(features[0], features[1])