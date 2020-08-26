import pickle

import torch

from autoencoder import *
from fastaHandler import *

path = "/home/roboticsloaner/Documents/projects/2020-science-fair-analysis/src/main/models/autoencoderModel(2).pkl"
with open(path, 'rb') as f:
    model = pickle.load(f)

#x = torch.FloatTensor([5 for i in range(566)])
handler = FASTAHandler()
handler.load("/home/roboticsloaner/Documents/projects/2020-science-fair-analysis/src/main/python/HA/2009.h3n2.fasta")
x = torch.FloatTensor(translateToNumerical(handler.loadSequence(handler.accessionList()[0])))

print(model.feature_vector_length)
output = model.forward(x)
rounded = torch.round(output)

print(model.current_feature_vector)

correct = (rounded == x).float().sum()
accuracy = correct / x.shape[0]
print(accuracy)