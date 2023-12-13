import torch
import torch.nn as nn


# class NeuralNetworkModel(nn.Module):
#     """
#     Nerual network model.
#     """
#     def __init__(
#         self, 
#         n_recordings: int,
#         n_neurons_per_recording: list, 
#         n_time_bins: int, 
#         width: int,
#         hidden_layers: int,
#         rank: int
#     ):
#         super(NeuralNetworkModel, self).__init__()
#         self.n_recordings = n_recordings
#         self.n_neurons_per_recording = n_neurons_per_recording
#         self.n_time_bins = n_time_bins
#         self.width = width
#         self.hidden_layers = hidden_layers
#         self.rank = rank
        
#         self.Us = [nn.Parameter(torch.randn(self.n_neurons_per_recording[i], self.rank)) for i in range(self.n_recordings)]
#         # self.V = nn.Parameter(torch.randn(self.rank, self.n_time_bins))
#         # self.feed_forward = nn.Sequential(
#         #     nn.Linear(self.rank * self.n_time_bins, self.width),
#         #     nn.ReLU(),
#         #     nn.Linear(self.width, self.width),
#         #     nn.ReLU(),
#         #     nn.Linear(self.width, self.n_time_bins)
#         # )
#         self.feed_forward = nn.Sequential(
#                 nn.Linear(self.rank * self.n_time_bins, self.width),
#                 nn.ReLU(),
#                 # Add dropout
#                 # nn.Dropout(p=0.5),
#                 *[nn.Linear(self.width, self.width) for _ in range(self.hidden_layers - 1)],
#                 nn.Linear(self.width, self.n_time_bins)
#             )
#         # self.biases = [nn.Parameter(torch.randn(self.n_neurons_per_recording[i], self.n_time_bins)) for i in range(self.n_recordings)]
#         self.bias = nn.Parameter(torch.randn(self.n_time_bins))

#     def forward(self, recording_index, recording_X):
#         # TODO: Check which dim to do average over
#         X = self.Us[recording_index].T @ recording_X
#         # print(X.shape)
#         X = torch.flatten(X, 1)
#         # print(X.shape)
#         Y_hat = self.feed_forward(X)
#         return Y_hat

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
        self.feed_forward = nn.Sequential(
                nn.Linear(self.rank * self.n_time_bins, self.width),
                nn.ReLU(),
                # Add dropout
                # nn.Dropout(p=0.5),
                *[nn.Linear(self.width, self.width) for _ in range(self.hidden_layers - 1)],
                nn.Linear(self.width, self.n_time_bins)
            )

    def forward(self, recording_index, recording_X):
        # TODO: Check which dim to do average over
        X = self.Us[recording_index].T @ recording_X
        # print(X.shape)
        X = torch.flatten(X, 1)
        # print(X.shape)
        Y_hat = self.feed_forward(X)
        return Y_hat
    
class NeuralNetworkClassificationModel(nn.Module):
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
        super(NeuralNetworkClassificationModel, self).__init__()
        self.n_recordings = n_recordings
        self.n_neurons_per_recording = n_neurons_per_recording
        self.n_time_bins = n_time_bins
        self.width = width
        self.hidden_layers = hidden_layers
        self.rank = rank
        
        self.Us = [nn.Parameter(torch.randn(self.n_neurons_per_recording[i], self.rank)) for i in range(self.n_recordings)]
        self.feed_forward = nn.Sequential(
                nn.Linear(self.rank * self.n_time_bins, self.width),
                nn.ReLU(),
                # Add dropout
                # nn.Dropout(p=0.5),
                *[nn.Linear(self.width, self.width) for _ in range(self.hidden_layers - 1)],
                nn.Linear(self.width, self.n_time_bins)
            )

    def forward(self, recording_index, recording_X):
        # TODO: Check which dim to do average over
        X = self.Us[recording_index].T @ recording_X
        # print(X.shape)
        X = torch.flatten(X, 1)
        # print(X.shape)
        Y_hat = self.feed_forward(X)
        # output a probability
        Y_hat = torch.sigmoid(Y_hat)
        return Y_hat