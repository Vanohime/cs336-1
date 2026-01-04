import torch
import torch.nn as nn
from torch.nn import init
from einops import rearrange, einsum
from math import sqrt

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        tensor = torch.empty((out_features, in_features), device=device, dtype=dtype)
        weights = nn.Parameter(tensor, requires_grad=True)
        with torch.no_grad():
            std = sqrt(2 / (in_features + out_features))
            init.trunc_normal_(weights, mean=0, std=std, a=-3*std, b=3*std)
        self.weights = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")

