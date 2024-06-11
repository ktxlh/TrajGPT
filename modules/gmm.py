from torch import nn


class GMM(nn.Module):
    """Predict GMM parameters given a representation vector"""
    def __init__(self, d_model, ncomp):
        super().__init__()
        self.ncomp = ncomp  # number of components in mixture
        self.d_model = d_model
        self.weight = nn.Sequential(
            nn.Linear(self.d_model, ncomp),
            nn.Softplus(),
        )
        self.loc = nn.Sequential(
            nn.Linear(self.d_model, ncomp),
        )
        self.scale = nn.Sequential(
            nn.Linear(self.d_model, ncomp),
            nn.Softplus(),
        )

    def forward(self, x, eps=1e-6):
        return {
            'weight': self.weight(x) + eps,
            'loc': self.loc(x),
            'scale': self.scale(x) + eps,
        }
