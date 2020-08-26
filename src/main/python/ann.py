import sys

from torch import nn
import torch

from timer import timed
from fastaHandler import *
from dataset import Dataset

class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # hyperparameters
        self.input_count = sys.argv[1]
        self.hidden_layers = sys.argv[2]
        self.hidden_layer_length = sys.argv[3]

        # define hidden layers
        self.layers = [self.input_layer] + [self.layer() for i in range(self.hidden_layers)] + [self.output_layer]

        self.sequential = nn.Sequential(*self.layers)

    @property
    def input_layer(self):
        return nn.Linear(self.input_count, self.hidden_layer_length)

    @property
    def output_layer(self):
        return nn.Linear(self.hidden_layer_length, self.input_count)

    def layer(self):
        return nn.Linear(self.hidden_layer_length, self.hidden_layer_length)

    def forward(self, x):
        x = self.sequential(x)
        return x

if __name__ == "__main__":
    @timed
    def run():
        assert len(sys.argv) >= 7, "Not enough command line arguments provided"

        network = Network()

        optimizer = torch.optim.Adam(network.parameters(), weight_decay=1e-5)

        criterion = nn.MSELoss()

        #training_data = [torch.FloatTensor([i for i in range(10)]) for i in range(1000)] # noisy training data
        handler = FASTAHandler()
        handler.load(f"/home/roboticsloaner/Documents/projects/2020-science-fair-analysis/src/main/python/HA/{sys.argv[4]}")
        training_data = [handler.loadSequence(accession) for accession in handler.accessionList()]
        #print(training_data)
        dts = Dataset(data=training_data)

        # train model
        num_epochs = int(sys.argv[5])
        batch_size = int(sys.argv[6])
        print(f"Training for {num_epochs} epochs...")
        batches = dts.batches(batch_size)
        train = batches[:-2]
        test = batches[-2:]
        losses = []
        losses_at_each_epoch = []
        for i in range(num_epochs):
            for batch in train:
                for x in batch:
                    #print(x)
                    optimizer.zero_grad()
                    output = network(x)
                    loss = criterion(output, x)
                    loss.backward()
                    losses.apend(loss.data)
                optimizer.step()
            print('epoch [{}/{}], loss: {:.4f}'.format(i + 1, num_epochs, loss.data))
            losses_at_each_epoch.append(loss)

        print("Training complete!")

        # test model
        sumAccuracy = 0
        tests = 0
        for batch in test:
            for x in batch:
                tests += 1
                output = network(x)
                correct = (output == x).float().sum()
                accuracy = correct / output.shape[0]
                sumAccuracy += accuracy
        print(f"Testing accuracy: {sumAccuracy / tests}")

        # save the file
        path = "/home/roboticsloaner/Documents/projects/2020-science-fair-analysis/src/main/models/ANNModel"
        suffix = ".pkl"
        i = 0
        while not os.path.exists(path + suffix):
            i += 1
            suffix = f"({i}).pkl"
        with open(path + suffix, "wb") as f:
            pickle.dump(network, f)

        # save loss graphs
        tail = "/home/roboticsloaner/Documents/projects/2020-science-fair-analysis/src/main/models/"

        head = "ANNEpochLosses"
        suffix = ".pkl"
        i = 0
        while not os.path.exists(tail+head+suffix):
            i += 1
            suffix = f"({i}).pkl"
        plt.plot(len(losses_at_each_epoch), losses_at_each_epoch)
        plt.savefig(tail+head+suffix)

        head = "ANNSampleLosses"
        suffix = ".pkl"
        i = 0
        while not os.path.exists(tail+head+suffix):
            i += 1
            suffix = f"({i}).pkl"
        plt.plot(len(losses), losses)
        plt.savefig(tail+head+suffix)