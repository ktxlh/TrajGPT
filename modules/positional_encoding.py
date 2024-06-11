# https://pytorch.org/tutorials/beginner/transformer_tutorial.html
import math

import torch
from torch import Tensor, nn


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))  # (d_model // 2,)
        pe = torch.zeros(max_len, d_model)  # (seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)  # (seq_len, d_model // 2)
        pe[:, 1::2] = torch.cos(position * div_term)  # (seq_len, d_model // 2)
        pe = pe.unsqueeze(0) # (1, seq_len, d_model)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, d_model]``
        """
        x += self.pe[:, :x.size(1)]  # (batch_size, seq_len, d_model)
        return self.dropout(x)