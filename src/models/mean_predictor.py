import torch
import torch.nn as nn

class MeanPredictor(torch.nn.Module):
    def __init__(self, Y_train):
        super(MeanPredictor, self).__init__()
        self.means = [torch.mean(torch.tensor(Y), dim=0) for Y in Y_train]

    def forward(self, recording_index, recording_X):
        return self.means[recording_index].repeat(recording_X.shape[0], 1)