import pickle

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

from fastaHandler import *

path = "/home/roboticsloaner/Documents/projects/2020-science-fair-analysis/src/main/python/HA/full.h3n2.fasta"
handler = FASTAHandler()
handler.load(path)

with open("differentIndices.pkl", "rb") as f:
    different_indices = pickle.load(f)

sequences = [translateToNumerical(handler.loadSequence(accession)) for accession in handler.accessionList()[:1000]]
relevant_sequences = []
for sequence in sequences:
    relevant_input = []
    for index in different_indices:
        relevant_input.append(sequence[index])
    relevant_sequences.append(relevant_input)

kmeans = KMeans(n_clusters=3).fit(relevant_sequences)
for centroid in kmeans.cluster_centers_:
    for val in centroid:
        print(val)
    print("\n")
plt.plot(kmeans.cluster_centers_, ".r")
plt.plot(relevant_sequences, ".b")
plt.show()