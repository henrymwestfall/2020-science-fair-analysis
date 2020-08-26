import pickle

from fastaHandler import *

path = '/home/roboticsloaner/Documents/projects/2020-science-fair-analysis/src/main/python/HA/full.h3n2.fasta'

handler = FASTAHandler()
handler.load(path)
sequences = [handler.loadSequence(accession) for accession in handler.accessionList()[:10000]]

similar_indices = [i for i in range(len(sequences[0]))]
similar_values = list(sequences[0])
doubts = [0 for value in similar_values]

for sequence in sequences:
    for index in similar_indices:
        if sequence[index] != similar_values[index]:
            doubts[index] += 1
different_indices = list(filter(lambda index: doubts[index] > 0.5 * len(sequences), similar_indices))
print(len(different_indices))

with open("differentIndices.pkl", "wb") as f:
    pickle.dump(different_indices, f)