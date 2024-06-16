from typing import List, Dict, Any, Tuple, Union
from collections import namedtuple
import torch
import copy
import numpy as np
import torch_geometric as pyg

from ding.torch_utils import Adam, to_device, to_dtype, unsqueeze, ContrastiveLoss
from ding.rl_utils import ppo_data, ppo_error, ppo_policy_error, ppo_policy_data, get_gae_with_default_last_value, \
    v_nstep_td_data, v_nstep_td_error, get_nstep_return_data, get_train_sample, gae, gae_data, ppo_error_continuous, \
    get_gae, ppo_policy_error_continuous
from ding.model import model_wrap

from alphazx.ppo.utils.model_wrapper import AlphaZXDistributionWrapper
from alphazx.ppo.utils.split_data_generator import split_data_generator
from ding.utils import POLICY_REGISTRY, RunningMeanStd
from ding.utils.data import default_collate, default_decollate
from ding.policy.base_policy import Policy
from ding.policy.common_utils import default_preprocess_learn

from alphazx.ppo.utils.learn_utils import alphazx_ppo_error, AlphaZXPPOData


@POLICY_REGISTRY.register('alphazx_ppo')
class AlphaZXPPOPolicy(Policy):
    """
    Overview:
        Policy class of on-policy version PPO algorithm. Paper link: https://arxiv.org/abs/1707.06347.
    """
    config = dict(
        # (str) RL policy register name (refer to function "POLICY_REGISTRY").
        type='ppo',
        # (bool) Whether to use cuda for network.
        cuda=False,
        # (bool) Whether the RL algorithm is on-policy or off-policy. (Note: in practice PPO can be off-policy used)
        on_policy=True,
        # (bool) Whether to use priority (priority sample, IS weight, update priority).
        priority=False,
        # (bool) Whether to use Importance Sampling Weight to correct biased update due to priority.
        # If True, priority must be True.
        priority_IS_weight=False,
        # (bool) Whether to recompurete advantages in each iteration of on-policy PPO.
        recompute_adv=True,
        # (bool) Whether to use nstep return to calculate value target, otherwise, use return = adv + value.
        nstep_return=False,
        # (bool) Whether to need policy ``_forward_collect`` output data in process transition.
        transition_with_policy_data=True,
        # learn_mode config
        learn=dict(
            # (int) After collecting n_sample/n_episode data, how many epoches to train models.
            # Each epoch means the one entire passing of training data.
            epoch_per_collect=10,
            # (int) How many samples in a training batch.
            batch_size=64,
            # (float) The step size of gradient descent.
            learning_rate=3e-4,
            # (float) The loss weight of value network, policy network weight is set to 1.
            value_weight=0.5,
            # (float) The loss weight of entropy regularization, policy network weight is set to 1.
            entropy_weight=0.0,
            # (float) PPO clip ratio, defaults to 0.2.
            clip_ratio=0.2,
            # (bool) Whether to use advantage norm in a whole training batch.
            adv_norm=True,
            # (bool) Whether to use value norm with running mean and std in the whole training process.
            value_norm=True,
            # (bool) Whether to enable special network parameters initialization scheme in PPO, such as orthogonal init.
            ppo_param_init=True,
            # (str) The gradient clip operation type used in PPO, ['clip_norm', clip_value', 'clip_momentum_norm'].
            grad_clip_type='clip_norm',
            # (float) The gradient clip target value used in PPO.
            # If ``grad_clip_type`` is 'clip_norm', then the maximum of gradient will be normalized to this value.
            grad_clip_value=0.5,
            # (bool) Whether ignore done (usually for max step termination env).
            ignore_done=False,
        ),
        # collect_mode config
        collect=dict(
            # (int) How many training samples collected in one collection procedure.
            # Only one of [n_sample, n_episode] should be set.
            # n_sample=64,
            # (int) Split episodes or trajectories into pieces with length `unroll_len`.
            unroll_len=1,
            # (float) Reward's future discount factor, aka. gamma.
            discount_factor=0.99,
            # (float) GAE lambda factor for the balance of bias and variance(1-step td and mc)
            gae_lambda=0.95,
        ),
        eval=dict(),  # for compatibility
    )

    def default_model(self) -> tuple[str, list[str]]:
        """
        Overview:
            Return this algorithm default neural network model setting for demonstration. ``__init__`` method will \
            automatically call this method to get the default model setting and create model.
        Returns:
            - model_info (:obj:`Tuple[str, List[str]]`): The registered model name and model's import_names.

        . note::
            The user can define and use customized network model but must obey the same inferface definition indicated \
            by import_names path. For example about PPO, its registered name is ``ppo`` and the import_names is \
            ``ppo.model.template.vac``.

        . note::
            Because now PPO supports both single-agent and multi-agent usages, so we can implement these functions \
            with the same policy and two different default models, which is controled by ``self._cfg.multi_agent``.
        """
        return 'alphazx_vac', ['alphazx.ppo.model.alphazx_vac']

    def _init_learn(self) -> None:
        """
        Overview:
            Initialize the learn mode of policy, including related attributes and modules. For PPO, it mainly contains \
            optimizer, algorithm-specific arguments such as loss weight, clip_ratio and recompute_adv. This method \
            also executes some special network initializations and prepares running mean/std monitor for value.
            This method will be called in ``__init__`` method if ``learn`` field is in ``enable_field``.

        . note::
            For the member variables that need to be saved and loaded, please refer to the ``_state_dict_learn`` \
            and ``_load_state_dict_learn`` methods.

        . note::
            For the member variables that need to be monitored, please refer to the ``_monitor_vars_learn`` method.

        . note::
            If you want to set some spacial member variables in ``_init_learn`` method, you'd better name them \
            with prefix ``_learn_`` to avoid conflict with other modes, such as ``self._learn_attr1``.
        """
        self._priority = self._cfg.priority
        self._priority_IS_weight = self._cfg.priority_IS_weight
        assert not self._priority and not self._priority_IS_weight, "Priority is not implemented in PPO"
        if self._cfg.learn.ppo_param_init:
            for n, m in self._model.named_modules():
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.orthogonal_(m.weight)
                    torch.nn.init.zeros_(m.bias)
        # Optimizer
        self._optimizer = Adam(
            self._model.parameters(),
            lr=self._cfg.learn.learning_rate,
            grad_clip_type=self._cfg.learn.grad_clip_type,
            clip_value=self._cfg.learn.grad_clip_value
        )

        self._learn_model = model_wrap(self._model, wrapper_name='base')

        # Algorithm config
        self._value_weight = self._cfg.learn.value_weight
        self._entropy_weight = self._cfg.learn.entropy_weight
        self._clip_ratio = self._cfg.learn.clip_ratio
        self._adv_norm = self._cfg.learn.adv_norm
        self._value_norm = self._cfg.learn.value_norm
        if self._value_norm:
            self._running_mean_std = RunningMeanStd(epsilon=1e-4, device=self._device)
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda
        self._recompute_adv = self._cfg.recompute_adv
        # Main model
        self._learn_model.reset()

    def _forward_learn(self, data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """
        Overview:
            Policy forward function of learn mode (training policy and updating parameters). Forward means \
            that the policy inputs some training batch data from the replay buffer and then returns the output \
            result, including various training information such as loss, clip_frac, approx_kl.
        Arguments:
            - data (:obj:`List[Dict[int, Any]]`): The input data used for policy forward, including the latest \
                collected training samples for on-policy algorithms like PPO. For each element in list, the key of the \
                dict is the name of data items and the value is the corresponding data. Usually, the value is \
                torch.Tensor or np.ndarray or there dict/list combinations. In the ``_forward_learn`` method, data \
                often need to first be stacked in the batch dimension by some utility functions such as \
                ``default_preprocess_learn``. \
                For PPO, each element in list is a dict containing at least the following keys: ``obs``, ``action``, \
                ``reward``, ``logit``, ``value``, ``done``. Sometimes, it also contains other keys such as ``weight``.
        Returns:
            - return_infos (:obj:`List[Dict[str, Any]]`): The information list that indicated training result, each \
                training iteration contains append a information dict into the final list. The list will be precessed \
                and recorded in text log and tensorboard. The value of the dict must be python scalar or a list of \
                scalars. For the detailed definition of the dict, refer to the code of ``_monitor_vars_learn`` method.

        .. tip::
            The training procedure of PPO is two for loops. The outer loop trains all the collected training samples \
            with ``epoch_per_collect`` epochs. The inner loop splits all the data into different mini-batch with \
            the length of ``batch_size``.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for PPOPolicy: ``ppo.policy.tests.test_ppo``.
        """
        data = default_preprocess_learn(data, ignore_done=self._cfg.learn.ignore_done, use_nstep=False)
        if self._cuda:
            data = to_device(data, self._device)
        data['obs'] = to_dtype(data['obs'], torch.float32)
        if 'next_obs' in data:
            data['next_obs'] = to_dtype(data['next_obs'], torch.float32)
        # ====================
        # PPO forward
        # ====================
        return_infos = []
        self._learn_model.train()

        for epoch in range(self._cfg.learn.epoch_per_collect):
            if self._recompute_adv:  # calculate new value using the new updated value network
                with torch.no_grad():
                    value = self._learn_model.forward(data['obs'], mode='compute_critic')['value']
                    next_value = self._learn_model.forward(data['next_obs'], mode='compute_critic')['value']
                    if self._value_norm:
                        value *= self._running_mean_std.std
                        next_value *= self._running_mean_std.std

                    traj_flag = data.get('traj_flag', None)  # traj_flag indicates termination of trajectory
                    compute_adv_data = gae_data(value, next_value, data['reward'], data['done'], traj_flag)
                    data['adv'] = gae(compute_adv_data, self._gamma, self._gae_lambda)

                    un_normalized_returns = value + data['adv']

                    if self._value_norm:
                        data['value'] = value / self._running_mean_std.std
                        data['return'] = un_normalized_returns / self._running_mean_std.std
                        self._running_mean_std.update(un_normalized_returns.cpu().numpy())
                    else:
                        data['value'] = value
                        data['return'] = un_normalized_returns

            else:  # don't recompute adv
                if self._value_norm:
                    un_normalized_return = data['adv'] + data['value'] * self._running_mean_std.std
                    data['return'] = un_normalized_return / self._running_mean_std.std
                    self._running_mean_std.update(un_normalized_return.cpu().numpy())
                else:
                    data['return'] = data['adv'] + data['value']

            for batch in split_data_generator(data, self._cfg.learn.batch_size, shuffle=True):
                output = self._learn_model.forward(batch['obs'], mode='compute_actor_critic')
                adv = batch['adv']
                if self._adv_norm:
                    # Normalize advantage in a train_batch
                    adv = (adv - adv.mean()) / (adv.std() + 1e-8)

                ppo_batch = AlphaZXPPOData(output['logit'], batch['logit'], batch['action'], output['value'],
                                           batch['value'], adv, batch['return'], batch['weight'])
                ppo_loss, ppo_info = alphazx_ppo_error(ppo_batch, self._clip_ratio)

                wv, we = self._value_weight, self._entropy_weight
                total_loss = ppo_loss.policy_loss + wv * ppo_loss.value_loss - we * ppo_loss.entropy_loss

                self._optimizer.zero_grad()
                total_loss.backward()
                self._optimizer.step()

                return_info = {
                    'cur_lr': self._optimizer.defaults['lr'],
                    'total_loss': total_loss.item(),
                    'policy_loss': ppo_loss.policy_loss.item(),
                    'value_loss': ppo_loss.value_loss.item(),
                    'entropy_loss': ppo_loss.entropy_loss.item(),
                    'adv_max': adv.max().item(),
                    'adv_mean': adv.mean().item(),
                    'value_mean': output['value'].mean().item(),
                    'value_max': output['value'].max().item(),
                    'approx_kl': ppo_info.approx_kl,
                    'clip_frac': ppo_info.clip_frac,
                }
                return_infos.append(return_info)
        return return_infos

    def _init_collect(self) -> None:
        """
        Overview:
            Initialize the collect mode of policy, including related attributes and modules. For PPO, it contains the \
            collect_model to balance the exploration and exploitation (e.g. the multinomial sample mechanism in \
            discrete action space), and other algorithm-specific arguments such as unroll_len and gae_lambda.
            This method will be called in ``__init__`` method if ``collect`` field is in ``enable_field``.

        .. note::
            If you want to set some spacial member variables in ``_init_collect`` method, you'd better name them \
            with prefix ``_collect_`` to avoid conflict with other modes, such as ``self._collect_attr1``.

        .. tip::
            Some variables need to initialize independently in different modes, such as gamma and gae_lambda in PPO. \
            This design is for the convenience of parallel execution of different policy modes.
        """
        self._unroll_len = self._cfg.collect.unroll_len
        self._collect_model = model_wrap(self._model, wrapper_name='alphazx_model_wrapper')
        self._collect_model.reset()
        self._gamma = self._cfg.collect.discount_factor
        self._gae_lambda = self._cfg.collect.gae_lambda
        self._recompute_adv = self._cfg.recompute_adv

    def _forward_collect(self, data: dict[int, Any]) -> dict[int, Any]:
        """
        Overview:
            Policy forward function of collect mode (collecting training data by interacting with envs). Forward means \
            that the policy gets some necessary data (mainly observation) from the envs and then returns the output \
            data, such as the action to interact with the envs.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action and \
                other necessary data (action logit and value) for learn mode defined in ``self._process_transition`` \
                method. The key of the dict is the same as the input data, i.e. environment id.

        . tip::
            If you want to add more tricks on this policy, like temperature factor in multinomial sample, you can pass \
            related data as extra keyword arguments of this method.

        . note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        . note::
            For more detailed examples, please refer to our unittest for PPOPolicy: ``ppo.policy.tests.test_ppo``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._collect_model.eval()
        with torch.no_grad():
            output = self._collect_model.forward(data, mode='compute_actor_critic')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _process_transition(self,
                            obs: pyg.data.Data,
                            policy_output: dict[str, torch.Tensor],
                            timestep: namedtuple) -> dict[str, torch.Tensor]:
        """
        Overview:
            Process and pack one timestep transition data into a dict, which can be directly used for training and \
            saved in replay buffer. For PPO, it contains obs, next_obs, action, reward, done, logit, value.
        Arguments:
            - obs (:obj:`torch.Tensor`): The env observation of current timestep, such as stacked 2D image in Atari.
            - policy_output (:obj:`Dict[str, torch.Tensor]`): The output of the policy network with the observation \
                as input. For PPO, it contains the state value, action and the logit of the action.
            - timestep (:obj:`namedtuple`): The execution result namedtuple returned by the environment step method, \
                except all the elements have been transformed into tensor data. Usually, it contains the next obs, \
                reward, done, info, etc.
        Returns:
            - transition (:obj:`Dict[str, torch.Tensor]`): The processed transition data of the current timestep.

        .. note::
            ``next_obs`` is used to calculate nstep return when necessary, so we place in into transition by default. \
            You can delete this field to save memory occupancy if you do not need nstep return.
        """
        transition = {
            'obs': obs,
            'next_obs': timestep.obs,
            'action': policy_output['action'],
            'logit': policy_output['logit'],
            'value': policy_output['value'],
            'reward': timestep.reward,
            'done': timestep.done,
        }
        return transition

    def _get_train_sample(self, transitions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Overview:
            For a given trajectory (transitions, a list of transition) data, process it into a list of sample that \
            can be used for training directly. In PPO, a train sample is a processed transition with new computed \
            ``traj_flag`` and ``adv`` field. This method is usually used in collectors to execute necessary \
            RL data preprocessing before training, which can help learner amortize revelant time consumption. \
            In addition, you can also implement this method as an identity function and do the data processing \
            in ``self._forward_learn`` method.
        Arguments:
            - transitions (:obj:`List[Dict[str, Any]`): The trajectory data (a list of transition), each element is \
                the same format as the return value of ``self._process_transition`` method.
        Returns:
            - samples (:obj:`List[Dict[str, Any]]`): The processed train samples, each element is the similar format \
                as input transitions, but may contain more data for training, such as GAE advantage.
        """
        data = transitions
        data = to_device(data, self._device)
        for transition in data:
            transition['traj_flag'] = copy.deepcopy(transition['done'])
        data[-1]['traj_flag'] = True

        if self._cfg.learn.ignore_done:
            data[-1]['done'] = False

        if data[-1]['done']:
            last_value = torch.zeros_like(data[-1]['value'])
        else:
            with torch.no_grad():
                last_value = self._collect_model.forward(
                    unsqueeze(data[-1]['next_obs'], 0), mode='compute_actor_critic'
                )['value']
            if len(last_value.shape) == 2:  # multi_agent case:
                last_value = last_value.squeeze(0)
        if self._value_norm:
            last_value *= self._running_mean_std.std
            for i in range(len(data)):
                data[i]['value'] *= self._running_mean_std.std
        data = get_gae(
            data,
            to_device(last_value, self._device),
            gamma=self._gamma,
            gae_lambda=self._gae_lambda,
            cuda=False,
        )
        if self._value_norm:
            for i in range(len(data)):
                data[i]['value'] /= self._running_mean_std.std

        # remove next_obs for save memory when not recompute adv
        if not self._recompute_adv:
            for i in range(len(data)):
                data[i].pop('next_obs')
        return get_train_sample(data, self._unroll_len)

    def _init_eval(self) -> None:
        """
        Overview:
            Initialize the eval mode of policy, including related attributes and modules. For PPO, it contains the \
            eval model to select optimal action (e.g. greedily select action with argmax mechanism in discrete action).
            This method will be called in ``__init__`` method if ``eval`` field is in ``enable_field``.
        """
        self._eval_model = model_wrap(self._model, wrapper_name='alphazx_model_wrapper')
        self._eval_model.reset()

    def _forward_eval(self, data: dict[int, Any]) -> dict[int, Any]:
        """
        Overview:
            Policy forward function of eval mode (evaluation policy performance by interacting with envs). Forward \
            means that the policy gets some necessary data (mainly observation) from the envs and then returns the \
            action to interact with the envs. ``_forward_eval`` in PPO often uses deterministic sample method to get \
            actions while ``_forward_collect`` usually uses stochastic sample method for balance exploration and \
            exploitation.
        Arguments:
            - data (:obj:`Dict[int, Any]`): The input data used for policy forward, including at least the obs. The \
                key of the dict is environment id and the value is the corresponding data of the env.
        Returns:
            - output (:obj:`Dict[int, Any]`): The output data of policy forward, including at least the action. The \
                key of the dict is the same as the input data, i.e. environment id.

        .. note::
            The input value can be torch.Tensor or dict/list combinations and current policy supports all of them. \
            For the data type that not supported, the main reason is that the corresponding model does not support it. \
            You can implement you own model rather than use the default model. For more information, please raise an \
            issue in GitHub repo and we will continue to follow up.

        .. note::
            For more detailed examples, please refer to our unittest for PPOPolicy: ``ppo.policy.tests.test_ppo``.
        """
        data_id = list(data.keys())
        data = default_collate(list(data.values()))
        if self._cuda:
            data = to_device(data, self._device)
        self._eval_model.eval()
        with torch.no_grad():
            output = self._eval_model.forward(data, mode='compute_actor')
        if self._cuda:
            output = to_device(output, 'cpu')
        output = default_decollate(output)
        return {i: d for i, d in zip(data_id, output)}

    def _monitor_vars_learn(self) -> list[str]:
        """
        Overview:
            Return the necessary keys for logging the return dict of ``self._forward_learn``. The logger module, such \
            as text logger, tensorboard logger, will use these keys to save the corresponding data.
        Returns:
            - necessary_keys (:obj:`List[str]`): The list of the necessary keys to be logged.
        """
        variables = super()._monitor_vars_learn() + [
            'policy_loss',
            'value_loss',
            'entropy_loss',
            'adv_max',
            'adv_mean',
            'approx_kl',
            'clip_frac',
            'value_max',
            'value_mean',
        ]
        return variables
