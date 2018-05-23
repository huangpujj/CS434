import os.path as path
import os
import numpy as np

EXAMPLES = 6000
DIMENSIS = 784

def kmeans(data, k=2, itr = 0):
    count = 0
    SSEs = [[] for i in range(k)]

    # Step 1 - Pick K random points as cluster centers called centroids.
    centroids = []
    orig_cntr = None
    for i in range(k):
        #centroids.append(data[np.random.randint(0, len(data))])
        centroids.append(data[i])

    # Step 2 - Repeat Step 3 and 4 until none of the cluster assignments change.
    while count != itr:
    #while not (np.array_equal(centroids, orig_cntr)):
        
        orig_cntr = centroids
        count += 1
        assignments = []        # Array of indexes

        new_centers = []

    # Step 3 - Assign each xi to nearest cluster by calculating its distance to each centroid.
        for j, x in enumerate(data):
            furthest = np.inf
            closest = 0
            for i, centroid in enumerate(centroids):
                dist = np.linalg.norm(x - centroid)
                if dist < furthest:
                    closest = i
                    furthest = dist
            assignments.append(closest)

    # Step 4 - Find new cluster center by taking the average of the assigned points.
        clusters = [[np.zeros(DIMENSIS), 0] for i in enumerate(centroids)]
        for i, a in enumerate(assignments):
            clusters[a][0] = np.add(clusters[a][0], data[i])    # Sum row
            clusters[a][1] += 1                                 # Total

        for i, row in enumerate(clusters):
            new_centers.append(np.divide(row[0], row[1]))       # Average
        
        centroids = new_centers

    # Calculate SSE
        values = [[] for i in range(k)]

        for row, assign in enumerate(assignments):
            values[assign].append(data[row])

        for i, value in enumerate(values):
            SSEs[i].append(np.sum((values[i] - centroids[i])**2))
            
    return SSEs

data = np.genfromtxt("data/data-1.txt", delimiter=',') # Get data, returns np array

if (data.shape != (EXAMPLES, DIMENSIS)): # Sanity check, should be (6000, 784)
    print("Error, incorrect dataset! Shape is " + str(data.shape) 
    + " but it should be " + str(EXAMPLES) + ", " + str(DIMENSIS))
    exit(1)

## Part 1
iterations = 30

totalSSEs = []

sse = kmeans(data, k = 2)

for i in range(len(sse[0])):
    tempVal = 0
    for j, SSE in enumerate(sse):
        tempVal += SSE[i]
    totalSSEs.append(tempVal)
print totalSSEs