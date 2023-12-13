import torch
import torch.nn as nn

class ZeroPredictor(torch.nn.Module):
    def __init__(self):
        super(ZeroPredictor, self).__init__()
        pass

    def forward(self, recording_index, recording_X):
        return torch.zeros((recording_X.shape[0], recording_X.shape[2]))