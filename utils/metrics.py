import torch
import torch.distributions as D
from torch.nn.functional import cross_entropy

from utils.constants import *
from utils.preprocess import compute_after_sep_mask


def instantiate_gmm(outputs, mask):
    """Instantiate a sequence of GMMs with torch.distributions

    outputs (dict): values Shape (batch_size, seq_len)
    mask (tensor) shape (batch_size, seq_len) or indices
    """
    weight, loc, scale = [outputs[k][mask] for k in ['weight', 'loc', 'scale']]  # (num_true_in_mask, num_gaussians)
    mix = D.Categorical(weight / weight.sum(dim=-1, keepdim=True))
    comp = D.Normal(loc, scale)
    gmm = D.mixture_same_family.MixtureSameFamily(mix, comp)
    return gmm


def compute_gmm_nll(outputs, labels, loss_mask):
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
    target_gmms = instantiate_gmm(outputs, loss_mask)
    target_labels = labels[loss_mask]
    nll = -target_gmms.log_prob(target_labels)
    return nll.mean()


def count_top_k_correct(outputs, labels):
    p = dict()
    _, topk_pred = torch.topk(outputs, k=max(TOP_KS), dim=-1)  # (# targets, k)
    topk_target = labels.unsqueeze(-1)  # (# targets, 1)
    for k in TOP_KS:
        # No need to use .any() because for prediction (not inference), at most one item is correct.
        p[k] = (topk_pred[..., :k] == topk_target).int().sum().item() / topk_target.shape[0]
    return p


def compute_p_within_t(outputs, labels, mask, support_upper):
    """
    outputs: dict GMM parameters -- unit: hour
    labels: (N, L) -- unit: hour
    mask: (N, L)
    support_upper: int -- unit: hour
    """
    target_labels = labels[mask]
    target_gmms = instantiate_gmm(outputs, mask)
    support_upper = torch.tensor(support_upper, device=labels.device, dtype=labels.dtype)
    support_lower = torch.zeros_like(target_labels)
    
    p = {}
    for minute_k in P_WITHIN_T:  # unit: minute
        hour_k = minute_k / 60
        
        # Clip distirbution
        interval_upper = target_labels + hour_k
        interval_upper[interval_upper > support_upper] = support_upper

        interval_lower = target_labels - hour_k
        interval_lower[interval_lower < 0] = 0

        # Re-normalize distribution
        nominator = (target_gmms.cdf(interval_upper) - target_gmms.cdf(interval_lower))
        denominator = (target_gmms.cdf(support_upper) - target_gmms.cdf(support_lower))
        p[minute_k] = (nominator / denominator).mean().item()

    return p


def compute_loss(output, target, num_regions):
    is_special = (target['region_id'] < N_SPECIAL_TOKENS)
    region_nll = cross_entropy(output['region_id'].reshape(-1, num_regions + N_SPECIAL_TOKENS), target["region_id"].reshape(-1), ignore_index=PAD)
    travel_nll = compute_gmm_nll(output['travel_time'], target['travel_time'], ~is_special)
    duration_nll = compute_gmm_nll(output['duration'], target['duration'], ~is_special)
    loss = region_nll + travel_nll + duration_nll
    return loss


def compute_scores(input, output, target, task, max_duration, max_travel_time):
    if task == NEXT_PREDICTION:
        metric_mask = torch.zeros_like(target['region_id'], dtype=torch.bool)
        metric_mask[:, -1] = True

    elif task == INFILLING:
        is_special = (target['region_id'] < N_SPECIAL_TOKENS)
        is_after_sep = compute_after_sep_mask(input['region_id'][:, 1:])
        metric_mask = (is_after_sep & (~is_special))

    # Do not predict actual missing spans between visits
    is_travel = (target['travel_time'] < MAX_VALID_TRAVEL_TIME)

    r_score = count_top_k_correct(output['region_id'][metric_mask], target['region_id'][metric_mask])
    d_score = compute_p_within_t(output['duration'], target['duration'], metric_mask, max_duration)
    t_score = compute_p_within_t(output['travel_time'], target['travel_time'], metric_mask & is_travel, max_travel_time)
    return {
        'region_id': r_score, 
        'travel_time': t_score,
        'duration': d_score, 
    }
