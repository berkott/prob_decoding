import numpy as np
import torch
import torch.nn as nn


class ReducedRankModel(nn.Module):
    """
    Better reduced rank model.
    """
    def __init__(
        self, 
        n_recordings: int,
        n_neurons_per_recording: list, 
        n_time_bins: int, 
        rank: int
    ):
        super(ReducedRankModel, self).__init__()
        self.n_recordings = n_recordings
        self.n_neurons_per_recording = n_neurons_per_recording
        self.rank = rank
        self.n_time_bins = n_time_bins
        
        self.Us = [nn.Parameter(torch.randn(self.n_neurons_per_recording[i], self.n_time_bins, self.rank))
                   for i in range(self.n_recordings)]
        self.V = nn.Parameter(torch.randn(self.rank, self.n_time_bins))
        # self.biases = [nn.Parameter(torch.randn(self.n_neurons_per_recording[i], self.n_time_bins)) for i in range(self.n_recordings)]
        # self.bias = nn.Parameter(torch.randn(self.n_time_bins))

    def forward(self, recording_index, recording_X):
        UV = np.einsum('ctr,rt->ctt', self.Us[recording_index], self.V)
        Y_hat = torch.einsum('ctt,kct->kt', UV, recording_X) # + self.bias
        return Y_hat