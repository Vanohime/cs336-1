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

class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        """
        Construct the RMSNorm module. This function should accept the following parameters:
        d_model: int Hidden dimension of the model
        eps: float = 1e-5 Epsilon value for numerical stability
        device: torch.device | None = None Device to store the parameters on
        dtype: torch.dtype | None = None Data type of the parameters
        """
        super().__init__()
        self.d_model: int = d_model
        self.eps: float = eps
        self.gain = nn.Parameter(torch.ones((d_model,), device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = einsum(x, x, "... d_model, ... d_model -> ...")
        rms = rms.unsqueeze(-1) # (..., 1)
        rms /= self.d_model
        rms += self.eps
        rms = torch.sqrt(rms)
        inv_rms = rms ** -1
        result = x * inv_rms
        result = einsum(result, self.gain, "... d_model, ... d_model -> ... d_model")
        return result.to(in_dtype)
        
class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w3 = Linear(in_features=d_model, out_features=d_ff, device=device, dtype=dtype)
        self.w2 = Linear(in_features=d_ff, out_features=d_model, device=device, dtype=dtype)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.w1(x) # (..., d_ff)
        z = y * torch.sigmoid(y) # (..., d_ff)
        t = self.w3(x) # (..., d_ff)
        return self.w2(z * t) #(..., d_model)

