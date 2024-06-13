import torch
import torch.nn as nn


class Space2Vec(nn.Module):
    """
    Vectorized implementation of https://arxiv.org/pdf/2003.00824#page=10.60
    """
    def __init__(self, d_embed, lambda_min, lambda_max, num_scales=64):
        super().__init__()

        self.lambda_min = lambda_min
        self.lambda_max = lambda_max
        self.g = lambda_max / lambda_min
        self.S = num_scales

        scales = torch.arange(num_scales).reshape(1, num_scales)
        self.register_buffer('scales', scales, persistent=False)

        a1 = torch.tensor([1, 0])
        a2 = torch.tensor([-1/2, torch.sqrt(torch.tensor(3))/2])
        a3 = torch.tensor([-1/2, -torch.sqrt(torch.tensor(3))/2])
        a = torch.stack([a1, a2, a3])  # (3, 2)
        self.register_buffer('a', a, persistent=False)

        self.location_embedding = nn.Sequential(
            nn.Linear(num_scales * 6, d_embed),
            nn.ReLU(),
        )


    def forward(self, x):
        """
        x:  (..., 2)
        Returns:  (..., d_embed)
        """
        nominator = (x @ self.a.T).unsqueeze(-1)  # (..., 3, 1)
        denominator = self.lambda_min * torch.pow(self.g, self.scales / self.S - 1)
        fraction = nominator / denominator  # (..., 3, num_scales)
        fraction = fraction.reshape(*fraction.shape[:-2], -1)  # (..., 3*num_scales)
        PE_sj = torch.concatenate([torch.cos(fraction), torch.sin(fraction)], axis=-1)  # (..., 6*num_scales)
        return self.location_embedding(PE_sj)  # (..., d_embed)
