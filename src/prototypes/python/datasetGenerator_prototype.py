import random
import pickle

class Pair(object):
    def __init__(self, _input, label, emphasis=1):
        """
        A class representing a data pair
        :param _input: the input to this pair
        :param label: the label for this pair
        """

        self._input = _input
        self.label = label
        self.emphasis = emphasis

    def __repr__(self):
        return f"({self._input},{self.label})x{self.emphasis}"


class Dataset(object):
    def __init__(self, data=None, source=""):
        """
        A dataset that stores training data and prepares training batches.
        :param source: the source pickle file containing the pairs of data points
        """

        #with open(source) as f:
        #    self.data = pickle.load(f)
        #    random.shuffle(self.data)
        if data == None:
            self.data = data#[Pair(random.randint(1, 5), random.randint(1, 5), random.randint(1, 10)) for i in range(10)]

    def batches(self, batchSize):
        batches = [self.data[batchNumber * batchSize:(batchNumber + 1) * batchSize] for batchNumber in range(len(self.data) // batchSize)]
        if len(batches) * batchSize < len(self.data):
            batches.append(self.data[batchSize * (len(batches) - 0):])
        return batches
