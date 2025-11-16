import torch
import torch.distributions as D
from torch.nn.functional import cross_entropy

from utils.constants import *
from utils.preprocess import compute_after_sep_mask


def instantiate_gmm(outputs, padding_mask):
    """Instantiate a sequence of GMMs with torch.distributions

    outputs (dict): values Shape (batch_size, seq_len)
    """
    weight, loc, scale = [
        outputs[k][~padding_mask] for k in ["weight", "loc", "scale"]
    ]  # (num_true_in_mask, num_gaussians)
    mix = D.Categorical(weight / weight.sum(dim=-1, keepdim=True))
    comp = D.Normal(loc, scale)
    gmm = D.mixture_same_family.MixtureSameFamily(mix, comp)
    return gmm


def compute_gmm_nll(outputs, labels, padding_mask):
    """
    Compute negative log likelihood of temporal labels given predicted GMM.

        Parameters:
            outputs (dict): {key (str): value (FloatTensor) Shape (batch_size, seq_len, ntoken, num_gaussians)}
                where key is one of {"weight", "loc", "scale"}
            labels (FloatTensor):                        Shape (batch_size, seq_len)
            loss_mask (BoolTensor):                      Shape (batch_size, seq_len)

        Returns:
            nll (FloatTensor): Negative log likelihood of shape (,)
    """
    target_gmms = instantiate_gmm(outputs, padding_mask)
    nll = -target_gmms.log_prob(labels[~padding_mask])
    return nll.mean()


def compute_loss(output, target):
    padding_mask = target["padding_mask"]
    place_nll = cross_entropy(
        output["place_id"][~padding_mask], target["place_id"][~padding_mask]
    )
    travel_nll = compute_gmm_nll(
        output["travel_time"], target["travel_time"], padding_mask
    )
    duration_nll = compute_gmm_nll(output["duration"], target["duration"], padding_mask)
    loss = place_nll + travel_nll + duration_nll
    return loss
