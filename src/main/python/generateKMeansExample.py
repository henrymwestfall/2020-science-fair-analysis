import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import random

k = 5
width = 1000
variance = 3
cluster_sizes = [random.randint(50, 100) for i in range(k)]
intended_centroids = [(random.randint(0, width), random.randint(0, width)) for i in range(k)]
points = []
for i, cluster in enumerate(cluster_sizes):
    centroid = intended_centroids[i]
    for i in range(cluster):
        p = (centroid[0] + random.randint(-variance * cluster, variance * cluster), centroid[1] + random.randint(-variance * cluster, variance * cluster))
        points.append(p)

kmeans = KMeans(k).fit(points)

points = dict(points)
centroids = dict(kmeans.cluster_centers_)

plt.plot(list(points.keys()), list(points.values()), ".b")
plt.plot(list(centroids.keys()), list(centroids.values()), ".r", markersize=10)
plt.show()
