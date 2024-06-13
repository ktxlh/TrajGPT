from torch import nn


class GMM(nn.Module):
    """Predict GMM parameters given a representation vector"""
    def __init__(self, d_model, num_gaussians):
        super().__init__()
        self.num_gaussians = num_gaussians  # number of components in mixture
        self.d_model = d_model
        self.weight = nn.Sequential(
            nn.Linear(self.d_model, num_gaussians),
            nn.Softplus(),
        )
        self.loc = nn.Sequential(
            nn.Linear(self.d_model, num_gaussians),
        )
        self.scale = nn.Sequential(
            nn.Linear(self.d_model, num_gaussians),
            nn.Softplus(),
        )

    def forward(self, x, eps=1e-6):
        return {
            'weight': self.weight(x) + eps,
            'loc': self.loc(x),
            'scale': self.scale(x) + eps,
        }
