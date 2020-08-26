# PyTorch practice
# autoencoder for 10 inputs
import pickle
import os
import sys

from torch import nn
import torch
import matplotlib.pyplot as plt

from dataset import Dataset
from fastaHandler import *
from timer import timed

class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # hyperparameters
        self.input_count = 20
        self.shrink_rate = float(sys.argv[2])
        self.non_feature_layers = int(sys.argv[1]) if len(sys.argv) == 2 else 10 # get layers from parameters
        self.encoder_hidden_node_counts = [int(self.input_count * self.shrink_rate ** i) for i in range(self.non_feature_layers // 2)]
        self.feature_vector_length = int(self.encoder_hidden_node_counts[-1] * self.shrink_rate)
        self.decoder_hidden_node_counts = [self.feature_vector_length] + list(reversed(self.encoder_hidden_node_counts))

        self.current_feature_vector = None

        # Define hidden layers
        encoder_layers, decoder_layers = list(map(self.encoder_layer, self.encoder_hidden_node_counts)), list(map(self.decoder_layer, self.decoder_hidden_node_counts))
        # Add activation functions
        insertions = 0
        for i in range(len(encoder_layers) * 2):
            if i % 2 == 1: # odd index
                encoder_layers.insert(i + insertions, nn.ReLU(True))
        for i in range(len(decoder_layers) * 2):
            if i % 2 == 1: # odd index
                decoder_layers.insert(i + insertions, nn.ReLU(True))

        # Define encoder and decoder
        self.encoder = nn.Sequential(*encoder_layers)
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        x = self.encoder(x)
        self.current_feature_vector = x.clone()
        x = self.decoder(x)
        return x

    def layer(self, input_node_count, section, section_output_length):
        count_index = section.index(input_node_count)
        if count_index + 1 < len(section):
            return nn.Linear(input_node_count, section[count_index + 1])
        else:
            return nn.Linear(input_node_count, section_output_length)

    def encoder_layer(self, input_node_count):
        return self.layer(input_node_count, self.encoder_hidden_node_counts, self.feature_vector_length)

    def decoder_layer(self, input_node_count):
        return self.layer(input_node_count, self.decoder_hidden_node_counts, self.input_count)

if __name__ == "__main__":
    @timed
    def run():
        assert len(sys.argv) >= 6, "Not enough command line arguments provided"

        network = Network()

        optimizer = torch.optim.Adam(network.parameters(), weight_decay=1e-5)

        criterion = nn.MSELoss()

        #training_data = [torch.FloatTensor([i for i in range(10)]) for i in range(1000)] # noisy training data
        handler = FASTAHandler()
        handler.load(f"/home/roboticsloaner/Documents/projects/2020-science-fair-analysis/src/main/python/HA/{sys.argv[3]}")
        training_data = [handler.loadSequence(accession) for accession in handler.accessionList()[:1000]]
        #print(training_data)
        dts = Dataset(data=training_data)
        prep = lambda sequence: translateToNumerical(clean(sequence))

        # similar indices
        with open("differentIndices.pkl", "rb") as f:
            different_indices = pickle.load(f)

        # train model
        num_epochs = int(sys.argv[4])
        batch_size = int(sys.argv[5])
        batches = dts.batches(batch_size, prep)
        train = batches[:len(batches) - 2]
        test = batches[len(batches) - 2:]
        losses = []
        losses_at_each_epoch = []
        print(f"Training for {num_epochs} epochs on {len(train)} batches (batch size {batch_size})...")
        for i in range(num_epochs):
            for batch in train:
                for x in batch:
                    optimizer.zero_grad()
                    relevant_input = []
                    for index in different_indices:
                        relevant_input.append(x[index].item())
                    relevant_input = torch.FloatTensor(relevant_input)
                    relevant_input.requires_grad_(True)

                    output = torch.round(network(relevant_input))
                    #print(relevant_input)
                    #print(x)
                    loss = criterion(output, relevant_input)
                    loss.backward()
                    losses.append(loss.data)
                optimizer.step()
            print('epoch [{}/{}], loss: {:.4f}'.format(i + 1, num_epochs, loss.data))
            losses_at_each_epoch.append(loss)

        print("Training complete!")

        # test model
        print(f"Testing model on {len(test)} batches")
        sum_accuracy = 0
        tests = 0
        for batch in test:
            for x in batch:
                tests += 1
                output = network(x)
                correct = (output == x).float().sum()
                accuracy = correct / output.shape[0]
                sum_accuracy += accuracy
        print(f"Testing accuracy: {sum_accuracy / tests}")

        # save the file
        path = "/home/roboticsloaner/Documents/projects/2020-science-fair-analysis/src/main/models/autoencoderModel"
        suffix = ".pkl"
        i = 0
        while os.path.exists(path + suffix):
            i += 1
            suffix = f"({i}).pkl"
        with open(path + suffix, "wb") as f:
            pickle.dump(network, f)

        # save loss graphsb
        tail = "/home/roboticsloaner/Documents/projects/2020-science-fair-analysis/src/main/graphs/"

        head = "autoencoderEpochLosses"
        suffix = ".png"
        i = 0
        while os.path.exists(tail+head+suffix):
            i += 1
            suffix = f"({i}).png"
        plt.plot(range(len(losses_at_each_epoch)), losses_at_each_epoch)
        plt.title("Loss vs Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.savefig(tail+head+suffix)
        plt.clf()

        head = "autoencoderSampleLosses"
        suffix = ".png"
        i = 0
        while os.path.exists(tail+head+suffix):
            i += 1
            suffix = f"({i}).png"
        plt.plot(range(len(losses)), losses)
        plt.title("Loss vs Iteration")
        plt.xlabel("Iteration")
        plt.ylabel("Loss")
        plt.savefig(tail+head+suffix)
        plt.clf()