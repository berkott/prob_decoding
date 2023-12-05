import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.metrics import accuracy_score, roc_auc_score, r2_score
from side_info_decoding.utils import sliding_window_over_trials

class Full_Rank_Model(nn.Module):
    """
    full rank model.
    """
    def __init__(
        self, 
        n_units, 
        n_t_bins, 
        
    ):
        super(Full_Rank_Model, self).__init__()
        self.n_units = n_units
        self.n_t_bins = n_t_bins
        self.window_size = 2*half_window_size+1
        self.Beta = nn.Parameter(torch.randn(self.n_units, self.n_t_bins, self.window_size))
        self.intercept = nn.Parameter(torch.randn((1,)))
        self.sigmoid = nn.Sigmoid()
        self.task_type = "single_task"
        self.model_type = "full_rank"

    def forward(self, X):
        n_trials = len(X)
        out = torch.einsum("ctd,kctd->k", self.Beta, X)
        out += self.intercept * torch.ones(n_trials)
        out = self.sigmoid(out)
        return out, self.Beta