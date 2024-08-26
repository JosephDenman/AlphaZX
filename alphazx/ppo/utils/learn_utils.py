from typing import NamedTuple, Optional

import torch

from alphazx.distributions import AlphaZXDistribution, AlphaZXDistributionParams


class AlphaZXPPOData(NamedTuple):
    logit_new: AlphaZXDistributionParams
    logit_old: AlphaZXDistributionParams
    # A batch of actions
    action: torch.Tensor
    value_new: torch.Tensor
    value_old: torch.Tensor
    adv: torch.Tensor
    return_: torch.Tensor
    weight: torch.Tensor


class AlphaZXPPOLoss(NamedTuple):
    policy_loss: torch.Tensor
    value_loss: torch.Tensor
    entropy_loss: torch.Tensor


class AlphaZXPPOPolicyInfo(NamedTuple):
    approx_kl: torch.Tensor
    clip_frac: torch.Tensor


def alphazx_ppo_error(
        data: AlphaZXPPOData,
        clip_ratio: float = 0.2,
        use_value_clip: bool = True,
        dual_clip: Optional[float] = None
) -> tuple[AlphaZXPPOLoss, AlphaZXPPOPolicyInfo]:
    raise NotImplementedError


class AlphaZXPPOPolicyData(NamedTuple):
    logit_new: AlphaZXDistributionParams
    logit_old: AlphaZXDistributionParams
    # A batch of actions
    action: torch.Tensor
    adv: torch.Tensor
    weight: torch.Tensor


class AlphaZXPPOPolicyLoss(NamedTuple):
    policy_loss: torch.Tensor
    entropy_loss: torch.Tensor


def alphazx_ppo_policy_error(data: AlphaZXPPOPolicyData,
                             clip_ratio: float = 0.2,
                             dual_clip: Optional[float] = None) -> tuple[AlphaZXPPOPolicyLoss, AlphaZXPPOPolicyInfo]:
    """
    Overview:
        Get PPO policy loss
    Arguments:
        - data: ppo input data with fieids shown in ``ppo_policy_data``
        - clip_ratio: clip value for ratio
        - dual_clip: a parameter c mentioned in arXiv:1912.09729 Equ. 5, shoule be in [1, inf),\
        defaults to 5.0, if you don't want to use it, set this parameter to None
    Returns:
        - ppo_policy_loss (:obj:`namedtuple`): the ppo policy loss item, all of them are the differentiable 0-dim tensor
        - ppo_info (:obj:`namedtuple`): the ppo optim information for monitoring, all of them are Python scalar
    Shapes:
        - logit_new (:obj:`torch.FloatTensor`): :math:`(B, N)`, where B is batch size and N is action dim
        - logit_old (:obj:`torch.FloatTensor`): :math:`(B, N)`
        - action (:obj:`torch.LongTensor`): :math:`(B, )`
        - adv (:obj:`torch.FloatTensor`): :math:`(B, )`
        - weight (:obj:`torch.FloatTensor` or :obj:`None`): :math:`(B, )`
        - policy_loss (:obj:`torch.FloatTensor`): :math:`()`, 0-dim tensor
        - entropy_loss (:obj:`torch.FloatTensor`): :math:`()`
    Examples:
    """
    logit_new, logit_old, action, adv, weight = data
    if weight is None:
        weight = torch.ones_like(adv)
    dist_new = AlphaZXDistribution(logit_new)
    dist_old = AlphaZXDistribution(logit_old)
    logp_new = dist_new.log_prob(action)
    logp_old = dist_old.log_prob(action)
    dist_new_entropy = dist_new.entropy()
    if dist_new_entropy.shape != weight.shape:
        dist_new_entropy = dist_new.entropy().mean(dim=1)
    entropy_loss = (dist_new_entropy * weight).mean()
    # policy_loss
    ratio = torch.exp(logp_new - logp_old)
    if ratio.shape != adv.shape:
        ratio = ratio.mean(dim=1)
    surrogate_1 = ratio * adv
    surrogate_2 = ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * adv
    if dual_clip is not None:
        clip1 = torch.min(surrogate_1, surrogate_2)
        clip2 = torch.max(clip1, dual_clip * adv)
        # only use dual_clip when adv < 0
        policy_loss = -(torch.where(adv < 0, clip2, clip1) * weight).mean()
    else:
        policy_loss = (-torch.min(surrogate_1, surrogate_2) * weight).mean()
    with torch.no_grad():
        approx_kl = (logp_old - logp_new).mean().item()
        clipped = ratio.gt(1 + clip_ratio) | ratio.lt(1 - clip_ratio)
        clip_frac = torch.as_tensor(clipped).float().mean().item()
    return AlphaZXPPOPolicyLoss(policy_loss, entropy_loss), AlphaZXPPOPolicyInfo(approx_kl, clip_frac)
