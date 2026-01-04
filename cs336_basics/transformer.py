import torch
import torch.nn as nn
from torch.nn import init
from einops import rearrange, einsum
from math import sqrt

class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        """
        Construct a linear transformation module. This function should accept the following parameters:
        in_features: int final dimension of the input
        out_features: int final dimension of the output
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        tensor = torch.empty((out_features, in_features), device=device, dtype=dtype)
        weights = nn.Parameter(tensor, requires_grad=True)
        with torch.no_grad():
            std = sqrt(2 / (in_features + out_features))
            init.trunc_normal_(weights, mean=0, std=std, a=-3*std, b=3*std)
        self.weights = weights

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weights, "... d_in, d_out d_in -> ... d_out")

class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        """
        Construct an embedding module. This function should accept the following parameters:
        num_embeddings: int Size of the vocabulary
        19
        embedding_dim: int Dimension of the embedding vectors, i.e., d_model
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        tensor = torch.empty((num_embeddings, embedding_dim), device=device, dtype=dtype)
        embedding_matrix = nn.Parameter(tensor, requires_grad=True)
        with torch.no_grad():
            init.trunc_normal_(embedding_matrix, mean=0, std=1, a=-3, b=3)
        self.embedding_matrix = embedding_matrix
    
    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.embedding_matrix[token_ids]

