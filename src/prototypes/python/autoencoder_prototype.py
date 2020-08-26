# PyTorch practice
# autoencoder for 10 inputs
from torch import nn
import torch
import random

class Network(nn.Module):
    def __init__(self):
        nn.Module.__init__(self)

        # hyperparameters
        self.input_count = 10
        self.non_feature_layers = 6
        self.encoder_hidden_node_counts = [int(self.input_count * 0.8 ** i) for i in range(self.non_feature_layers // 2)]
        self.feature_vector_length = int(self.encoder_hidden_node_counts[-1] * 0.8)
        self.decoder_hidden_node_counts = [self.feature_vector_length] + list(reversed(self.encoder_hidden_node_counts))

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

network = Network()

x = torch.FloatTensor([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

output = network.forward(x)
target = torch.randn(10)
#target = target.view(1, -1)
criterion = nn.MSELoss()

loss = criterion(output, target)

optimizer = torch.optim.Adam(network.parameters(), weight_decay=1e-5)

training_data = [torch.FloatTensor([i for i in range(10)]) for i in range(1000)] # noisy training data

num_epochs = 10
epoch = 100
for i in range(num_epochs * epoch):
    x = training_data[i]
    output = network(x)
    loss = criterion(output, x)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if i % epoch == 0:
        print('epoch [{}/{}], loss:{:.4f}'.format((i // epoch) + 1, num_epochs, loss.data))