from typing import Optional

import numpy as np
import torch
import torch.nn as nn


class SigmoidCrossAttention(nn.Module):
    """
    Args: dim, mask
        dim (int): dimension of attention
        mask (torch.Tensor): tensor containing indices to be masked

    Inputs: query, key, value, mask
        - **query** (batch, q_len, d_model): tensor containing projection vector for decoder.
        - **key** (batch, k_len, d_model): tensor containing projection vector for encoder.
        - **mask** (-): tensor containing indices to be masked

    Returns: attn
        - **attn**: tensor containing the attention (alignment) from the encoder outputs.
    """
    def __init__(self, dim: int):
        super(SigmoidCrossAttention, self).__init__()
        self.q_lin = nn.Linear(dim, dim)
        self.k_lin = nn.Linear(dim, dim)
        self.sqrt_dim = np.sqrt(dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        query_proj = self.q_lin(query)
        key_proj = self.k_lin(key)
        score = torch.bmm(query_proj, key_proj.transpose(1, 2)) / self.sqrt_dim
        if mask is not None:
            score.masked_fill_(mask.view(score.size()), -float('Inf'))
        return torch.sigmoid(score)
