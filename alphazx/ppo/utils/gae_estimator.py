from typing import TYPE_CHECKING, Callable, Optional, Any
from easydict import EasyDict

import torch
import torch_geometric as pyg
import treetensor.torch as ttorch
from ding.policy import Policy
from ding.data import Buffer
from ding.rl_utils import gae, gae_data, get_train_sample
from ding.framework import task
from ding.utils.data import ttorch_collate
from ding.torch_utils import to_device


def is_batched_graph(data: Any) -> bool:
    assert isinstance(data, pyg.data.Data), f"Expected data value {data} to be a PyG graph."
    assert 'batch' in data, f"Expected data object {data} to contain 'batch' key."
    return torch.all(data.batch == 0).item()


def gae_estimator(cfg: EasyDict, policy: Policy, buffer_: Optional[Buffer] = None) -> Callable:
    """
    Overview:
        Calculate value using observation of input data, then call function `gae` to get advantage. \
        The processed data will be pushed into `buffer_` if `buffer_` is not None, \
        otherwise it will be assigned to `ctx.train_data`.
    Arguments:
        - cfg (:obj:`EasyDict`): Config which should contain the following keys: \
            `cfg.policy.collect.discount_factor`, `cfg.policy.collect.gae_lambda`.
        - policy (:obj:`Policy`): Policy in `policy.collect_mode`, used to get model to calculate value.
        - buffer\_ (:obj:`Optional[Buffer]`): The `buffer_` to push the processed data in if `buffer_` is not None.
    """
    if task.router.is_active and not task.has_role(task.role.LEARNER):
        return task.void()

    model = policy.get_attribute('model')
    # Unify the shape of obs and action

    def _gae(ctx: "OnlineRLContext"):
        """
        Input of ctx:
            - trajectories (:obj:`List[treetensor.torch.Tensor]`): The data to be processed.\
                Each element should contain the following keys: `obs`, `next_obs`, `reward`, `done`.
            - trajectory_end_idx: (:obj:`treetensor.torch.IntTensor`):
                The indices that define the end of trajectories, \
                which should be shorter than the length of `ctx.trajectories`.
        Output of ctx:
            - train_data (:obj:`List[treetensor.torch.Tensor]`): The processed data if `buffer_` is None.
        """
        cuda = cfg.policy.cuda and torch.cuda.is_available()

        # action shape (B,) for discete action, (B, D,) for continuous action
        # reward shape (B,) done shape (B,) value shape (B,)
        print('ctx.trajectories = ', ctx.trajectories)
        data = ttorch_collate(ctx.trajectories, cat_1dim=True)
        if data['action'].dtype in [torch.float16, torch.float32, torch.double] \
                and data['action'].dim() == 1:
            # action shape
            data['action'] = data['action'].unsqueeze(-1)

        with torch.no_grad():
            if cuda:
                data = data.cuda()
            value = model.forward(data.obs.to(dtype=ttorch.float32), mode='compute_critic')['value']
            next_value = model.forward(data.next_obs.to(dtype=ttorch.float32), mode='compute_critic')['value']
            data.value = value

            traj_flag = data.done.clone()
            traj_flag[ctx.trajectory_end_idx] = True
            data.traj_flag = traj_flag

            # done is bool type when acquired from env.step
            data_ = gae_data(data.value, next_value, data.reward, data.done.float(), traj_flag.float())
            data.adv = gae(data_, cfg.policy.collect.discount_factor, cfg.policy.collect.gae_lambda)
        if buffer_ is None:
            ctx.train_data = data
        else:
            data = data.cpu()
            data = ttorch.split(data, 1)
            # To ensure the shape of obs is same as config
            if is_batched_graph(data[0]['obs']):
                if 'logit' in data[0]:
                    for d in data:
                        d['logit'] = d['logit'].squeeze(0)
                if 'log_prob' in data[0]:
                    for d in data:
                        d['log_prob'] = d['log_prob'].squeeze(0)
            else:
                pass

            if data[0]['action'].dtype in [torch.float16, torch.float32, torch.double] \
                    and data[0]['action'].dim() == 2:
                for d in data:
                    d['action'] = d['action'].squeeze(0)
            for d in data:
                buffer_.push(d)
        ctx.trajectories = None

    return _gae