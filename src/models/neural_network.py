import torch
import torch.nn as nn


class NeuralNetworkModel(nn.Module):
    """
    Nerual network model.
    """
    def __init__(
        self, 
        n_recordings: int,
        n_neurons_per_recording: list, 
        n_time_bins: int, 
        width: int,
        hidden_layers: int,
        rank: int
    ):
        super(NeuralNetworkModel, self).__init__()
        self.n_recordings = n_recordings
        self.n_neurons_per_recording = n_neurons_per_recording
        self.n_time_bins = n_time_bins
        self.width = width
        self.hidden_layers = hidden_layers
        self.rank = rank
        
        self.Us = [nn.Parameter(torch.randn(self.n_neurons_per_recording[i], self.rank)) for i in range(self.n_recordings)]
        self.biases = [nn.Parameter(torch.randn(self.rank, self.n_time_bins)) for i in range(self.n_recordings)]

        modules = [
            nn.Linear(self.rank * self.n_time_bins, self.width),
            nn.ReLU()
        ]
        for _ in range(self.hidden_layers - 1):
            modules.append(nn.Linear(self.width, self.width))
            modules.append(nn.ReLU())
        modules.append(nn.Linear(self.width, self.n_time_bins))
        self.feed_forward = nn.Sequential(*modules)

    def forward(self, recording_index, recording_X):
        X = self.Us[recording_index].T @ recording_X + self.biases[recording_index]
        X = torch.flatten(X, 1)
        Y_hat = self.feed_forward(X)
        return Y_hat
    