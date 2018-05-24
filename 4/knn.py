import os.path as path
import os
import numpy as np

EXAMPLES = 6000
DIMENSIS = 784

def loop(count, centroids, orig_cntr, itr=0):
    if itr != None:
        return count != itr
    return not np.array_equal(centroids, orig_cntr)

def kmeans(data, k=2, itr = None):
    count = 0
    orig_cntr = None
    SSEs = [[] for i in range(k)]
    centroids = []

    # Step 1 - Pick K random points as cluster centers called centroids.
    for i in range(k):
        centroids.append(data[np.random.randint(0, len(data))])    # Random centers
        #centroids.append(data[i])                                 # Sanity check

    # Step 2 - Repeat Step 3 and 4 until none of the cluster assignments change.
    while loop(count, centroids, orig_cntr, itr):
        orig_cntr = centroids
        count += 1
        assignments = []                                            # Array of indexes
        predicted = [[] for i in range(k)]

    # Step 3 - Assign each xi to nearest cluster by calculating its distance to each centroid.
        for j, x in enumerate(data):
            furthest = np.inf
            closest = 0
            for i, centroid in enumerate(centroids):
                dist = np.sqrt( np.sum((x - centroid)**2) )
                if dist < furthest:
                    furthest = dist
                    closest = i
            assignments.append(closest)

    # Step 4 - Find new cluster center by taking the average of the assigned points.
        clusters = [[np.zeros(DIMENSIS), 0] for i in enumerate(centroids)]
        for i, a in enumerate(assignments):
            clusters[a][0] = np.add(clusters[a][0], data[i])    # Sum row
            clusters[a][1] += 1                                 # Total
        
        centroids = []                                          # Reset
        for i, row in enumerate(clusters):
            centroids.append(np.divide(row[0], row[1]))         # Avg

    # Calculate SSE
        for row, a in enumerate(assignments):
            predicted[a].append(data[row])

        for i, yi in enumerate(predicted):
            SSEs[i].append(np.sum((yi - centroids[i])**2))
    
    collated = []
    for i in range(len(SSEs[0])):
        x = 0
        for j, s in enumerate(SSEs):
            x += s[i]
        collated.append(x)

    return collated


def part2_1(data, iterations):
    sse = kmeans(data, k = 2, itr = iterations)
    
    part2_1 = open("part2_1.csv", "w+")
    
    print "Iteration\tSSE"
    
    for i, j in enumerate(sse):
        print(str(i+1) + "\t" + str(j))
        part2_1.write(str(i+1) + "," + str(j) + "\n")
    
    part2_1.close()


# Main

data = np.genfromtxt("data/data-1.txt", delimiter=',')              # Get data, returns np array

if (data.shape != (EXAMPLES, DIMENSIS)):                            # Sanity check, should be (6000, 784)
    print("Error, incorrect dataset! Shape is " + str(data.shape) 
    + " but it should be " + str(EXAMPLES) + ", " + str(DIMENSIS))
    exit(1)

## Non-hierarchical clustering - K-Means algorithm
part2_1(data, 30)