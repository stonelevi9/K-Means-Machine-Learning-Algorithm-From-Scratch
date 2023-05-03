import pandas as pd
import numpy as np
import random
from math import sqrt
from sklearn import preprocessing
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import matplotlib.patches as patch


# This is our initialize method, and it is responsible for preparing and processing our data set to be worked with
def initialize():
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    data = np.delete(data, [8], axis=1)
    data = preprocessing.normalize(data)
    return data, target


# This is our initial_centroids method, and it is responsible for creating our initial centroids. It does this by
# picking random indexes of our data set and setting them as our initial centroids.
def initial_centroids(data, k):
    centroids = np.empty([k, 12])
    for i in range(k):
        x = random.randint(0, 505)
        centroids[i] = data[x]
    return centroids


# This is our k_means method, and it is responsible for applying the k_means algorithm to our dataset. It works by
# calling our initial_centroids method to initialize our centroids, and then it enters a while loop that runs until
# our previously calculated centroid is equal to our current centroids (or until they do not change). It calls our
# cluster assignment method to perform most of the logic as well as compute_new_centroid to figure out our new
# centroids.
def k_means(data, k):
    centroids = initial_centroids(data, k)
    prev_cent = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]],
                         )
    count = 0
    while not (centroids == prev_cent).all():
        count += 1
        print('Run number ', count)
        clusters = cluster_assignment(data, centroids)
        prev_cent = np.copy(centroids)
        centroids = np.copy(compute_new_centroid(data, clusters, centroids))

    print('centroids:', centroids)
    print('prev_cent:', prev_cent)
    return clusters


# This is our compute_new_centroid method, and it is responsible for calculating and updating our new centroids. It
# works by iterating through each centroid row and comparing it to the current centroid's value (0-3) to determine an
# average of all the points assigned to a specific centroid, and then it returns newly computed centroids based on these
# averages.
def compute_new_centroid(data, clusters, centroids):
    for i in range(4):
        count = 0
        new_centroid = np.empty(12)
        for j in range(506):
            if clusters[j] == i:
                count += 1
                for h in range(12):
                    new_centroid[h] = new_centroid[h] + data[j][h]
        for p in range(12):
            if count != 0:
                new_centroid[p] = new_centroid[p] / count
        centroids[i] = new_centroid
    return centroids


# This is our cluster_assignment method, and it is responsible for calling our cluster_calc and find_closest_cluster
# method on each of our data elements to determine it's closest cluster.
def cluster_assignment(data, centroids):
    clusters = np.empty(506)
    for i in range(506):
        cluster0 = cluster_calc(data[i], centroids[0])
        cluster1 = cluster_calc(data[i], centroids[1])
        cluster2 = cluster_calc(data[i], centroids[2])
        cluster3 = cluster_calc(data[i], centroids[3])
        current_cluster = find_closest_cluster(cluster0, cluster1, cluster2, cluster3)
        clusters[i] = current_cluster
    return clusters


# This is our cluster_calc method, and it is responsible for computing the distance of a data entry to a specific
# centroid and returning that value.
def cluster_calc(data, centroids):
    difference = np.subtract(data, centroids)
    squared = np.square(difference)
    ones_matrix = np.ones(len(squared))
    dot_product = np.dot(squared, ones_matrix)
    return sqrt(dot_product)


# This is our find_closest_cluster method, and it is responsible for finding the closest cluster given the distance of
# a point from all the clusters.
def find_closest_cluster(cluster0, cluster1, cluster2, cluster3):
    current_cluster = cluster0
    cluster_num = 0
    if cluster1 <= current_cluster:
        current_cluster = cluster1
        cluster_num = 1
    if cluster2 <= current_cluster:
        current_cluster = cluster2
        cluster_num = 2
    if cluster3 <= current_cluster:
        cluster_num = 3
    return cluster_num


# This is our visualize method, and it is responsible for using TSNE to reduce the dimensions of our dataset so that
# we can plot and visualize our clusters.
def visualize(new_data, clusters):
    tsne = TSNE(n_components=2, n_iter=5000, random_state=0)
    tsne_output = tsne.fit_transform(new_data)
    tsne_x = tsne_output[:, 0]
    tsne_y = tsne_output[:, 1]
    colors = {0: 'yellow', 1: 'red', 2: 'blue', 3: 'green'}
    for i in range(len(clusters)):
        plt.scatter(tsne_x[i], tsne_y[i], color=colors.get(clusters[i], 'white'))
    cluster1 = patch.Patch(color='yellow', label='Cluster 1')
    cluster2 = patch.Patch(color='red', label='Cluster 2')
    cluster3 = patch.Patch(color='blue', label='Cluster 3')
    cluster4 = patch.Patch(color='green', label='Cluster 4')
    plt.legend(handles=[cluster1, cluster2, cluster3, cluster4])
    plt.show()

# This is our main method. It initializes our dataset, then calls k_means, then calls our visualize method.
def main():
    data, target = initialize()
    clusters = k_means(data, 4)
    visualize(data, clusters)


if __name__ == '__main__':
    main()
