import torch
import torch.nn as nn
from torch.nn import init
from einops import rearrange, einsum
from math import sqrt, sin, cos
import einx

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

class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        """
        Construct the RoPE module and create buffers if needed.
        theta: float Θ value for the RoPE
        d_k: int dimension of query and key vectors
        max_seq_len: int Maximum sequence length that will be inputted
        device: torch.device | None = None Device to store the buffer on
        """
        super().__init__()
        i_range = torch.arange(max_seq_len, device=device)
        k_range = torch.arange(1,  d_k // 2 + 1, device=device)
        i_grid, k_grid =  torch.meshgrid(i_range, k_range, indexing='ij') # (seq_len, d_k //2)
        theta_tensor = torch.tensor(theta, device=device)
        theta_i_k = i_grid / torch.pow(theta_tensor, (2 * k_grid - 2) / d_k) # (seq_len, d_k //2)
        cos_ik = torch.cos(theta_i_k)
        sin_ik = torch.sin(theta_i_k)
        #Here we transpose the matrix, as torch uses row vectors, not column vectors
        # TODO: use only einops.rearrange later
        r_0 = torch.stack((cos_ik, sin_ik), dim=-1) # (seq_len, d_k //2, 2) 
        r_1 = torch.stack((-sin_ik, cos_ik), dim=-1) # (seq_len, d_k //2, 2)
        R = torch.stack((r_0, r_1), dim=-1) # (seq_len, d_k //2, 2, 2)
        self.register_buffer("R", R, persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        R = self.R[token_positions] # (..., num_tokens, d_k //2, 2, 2)
        x_reshaped = rearrange(
        x, 
        "... num_tokens (d_half d_pair) -> ... num_tokens d_half d_pair",
        d_pair=2
        )
        res = einsum(
            x_reshaped, R,
            "... num_tokens d_half d_pair_in, ... num_tokens d_half d_pair_out d_pair_in -> ... num_tokens d_half d_pair_out"
        )
        res = rearrange(res, "... num_tokens d_half d_pair -> ... num_tokens (d_half d_pair)", d_pair=2)
        return res

def softmax(x: torch.Tensor, i: int) -> torch.Tensor:
    x_norm = torch.exp(x - torch.max(x, dim=i, keepdim=True).values)
    sums = torch.sum(x_norm, dim=i, keepdim=True)
    return x_norm / sums