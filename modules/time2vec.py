import torch
import torch.nn as nn

class Time2Vec(nn.Module):
    """
    Implementation of https://arxiv.org/pdf/1907.05321.pdf
    """
    def __init__(self, d_embed):
        super().__init__()
        self.d_embed = d_embed
        self.linear = nn.Linear(1, d_embed)
        self.act = torch.sin

    def forward(self, x):
        """
        x: N, L
        o: N, L, d_embed
        """
        x = x.unsqueeze(-1)  # N, L, 1
        h = self.linear(x)  # N, L, d_embed
        # Inplace operation causes RuntimeError for gradient computation. Concat is necessary.
        o = torch.concat([h[..., :1], self.act(h[..., 1:])], axis=-1)
        return o
