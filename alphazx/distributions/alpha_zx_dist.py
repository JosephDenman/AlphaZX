from typing import Literal, NamedTuple

import torch
from torch.distributions.categorical import Categorical

from alphazx.distributions.bernoulli_mixture import MultivariateBernoulli
from alphazx.models import assert_not_all_zero

torch.set_printoptions(threshold=60_000)

def safe_log(t: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(t.dtype).eps
    return torch.log(torch.clamp(t, min=eps))


def check_non_zero_elems_exist(t: torch.Tensor) -> None:
    num_zero_elems = torch.sum((t == 0.))
    num_elems = t.numel()
    assert num_zero_elems != num_elems, f'All elements of {t} are zero'


class AlphaZXDistributionParams(NamedTuple):
    mixture_dist_probs: torch.Tensor
    node_dist_probs: torch.Tensor
    phase_dist_probs: torch.Tensor
    new_edge_dist_probs: torch.Tensor
    transfer_edge_dist_probs: torch.Tensor


class AlphaZXDistribution:
    """
    Currently, limited to two rewrite types: frz and flz, represented as 0 and 1 in the first entry of an action tensor.
    In the following:

        B = batch size
        K = number of samples
        L = length of the longest action (either in the batch or sample set)
        T = number of possible rewrites (frz, flz, frx, flx, etc.)
        N = max number of nodes across node types across batches, or across samples
        P = number of phase buckets
        E_new = number of new edge buckets
        E_trans = max degree of any node in the batch

    For simplicity, T is always fixed to the maximum (two in this case).
    """

    def __init__(self, params: AlphaZXDistributionParams, device: torch.device = 'cpu'):
        """
        :param params: A dictionary of tensors representing the parameters of the distribution. It contains the following keys:
        mixture_dist_params: B x T tensor of mixture probabilities. mixture_dist_params[b] is the mixture probabilities
                             at some step in a trajectory. It is assumed to be a softmax output of a DNN. When B = 1,
                             we are in the MCTS portion of the algorithm.
        node_dist_params: B x T x N tensor of node selection probabilities. For a node n that is not of type t in batch b,
                          node_dist_params[b, t, n] = 0.
        phase_dist_params: B x N x P tensor of phase probabilities. For a node n in batch b that does not
                           represent an f-right match, phase_dist_params[b, n] = torch.zeroes((P, ))
        new_edge_dist_params: B x N x E_new tensor of new edge probabilities. For a node n in batch b that does not
                              represent an f-right match, new_edges_dist_params[b, n] is all zeros.
        transfer_edge_dist_params: B x N x E_trans tensor of probabilities for each node. For a node n in batch b
                                   that does not represent an f-right match, transfer_edges_dist_params[b, n] has a single 1 entry
                                   in the top left corner of the innermost 2D tensor. All other entries in the innermost 2D tensor are 0.
                                   Samples drawn from this distribution are all zeros (no edges are selected to be transferred).
        """
        self.mixture_dist_params = params.mixture_dist_probs.to(device)
        check_non_zero_elems_exist(params.node_dist_probs)
        self.node_dist_params = params.node_dist_probs.to(device)
        self.phase_dist_params = params.phase_dist_probs.to(device)
        self.new_edge_dist_params = params.new_edge_dist_probs.to(device)
        self.transfer_edge_dist_params = params.transfer_edge_dist_probs.to(device)

    def _sample_action_types(self, k: int) -> torch.Tensor:
        return Categorical(probs=self.mixture_dist_params).sample(torch.Size([k])).T

    def _action_type_log_probs(self, action_types: torch.Tensor) -> torch.Tensor:
        return torch.gather(safe_log(self.mixture_dist_params), 1, action_types)

    def _select_node_dist_params(self, action_types: torch.Tensor) -> torch.Tensor:
        # Reshape action_types for broadcasting over the distributions
        action_types_expanded = action_types.unsqueeze(-1).expand(-1, -1, self.node_dist_params.size(-1))
        selected_node_dist_params = torch.gather(self.node_dist_params, 1, action_types_expanded)
        # print('action_types = ', action_types)
        # print('selected_node_dist_params = ', selected_node_dist_params)
        return selected_node_dist_params

    @staticmethod
    def _sample_from_selected_dist_params(selected_dist_params: torch.Tensor) -> torch.Tensor:
        # Flatten the tensor from [batch_size, num_distributions, num_classes] to [batch_size * num_distributions, num_classes]
        # because categorical treats the first dimension as the batch dimension
        flattened_distributions = selected_dist_params.view(-1, selected_dist_params.size(-1))
        dist = Categorical(probs=flattened_distributions)
        samples = dist.sample()
        # Reshape the samples back to the original [batch_size, num_distributions] format
        reshaped_samples = samples.view(selected_dist_params.size(0), selected_dist_params.size(1))
        return reshaped_samples

    def _sample_nodes(self, action_types: torch.Tensor) -> torch.Tensor:
        return self._sample_from_selected_dist_params(self._select_node_dist_params(action_types))

    def _node_log_probs(self, action_types: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        try:
            selected_node_dist_params = self._select_node_dist_params(action_types)
            assert_not_all_zero(selected_node_dist_params)
            batch_size, num_nodes = nodes.shape
            # Generate a tensor of batch indices to pair with each node index
            batch_indices = torch.arange(batch_size).view(-1, 1).expand_as(nodes)
            # Select the distribution parameters for each node
            node_log_probs = safe_log(selected_node_dist_params)[batch_indices, torch.arange(num_nodes), nodes]
            return node_log_probs
        except IndexError as error:
            # print('node_dist_params = ', self.node_dist_params)
            # print('action_types = ', action_types)
            # print('nodes = ', nodes)
            # print('nodes.shape = ', nodes.shape)
            # print('selected_feature_dist_params = ', selected_node_dist_params)
            # print('batch_indices = ', batch_indices)
            # print('torch.arange(num_nodes) = ', torch.arange(num_nodes))
            raise error

    def _select_feature_dist_params(self,
                                    nodes: torch.Tensor,
                                    feature_type: Literal['phase'] | Literal['new_edge'] | Literal[
                                        'transfer_edge']) -> torch.Tensor:
        """
        Selects rows from feature distribution parameters based on indices in nodes.

        :param nodes: A tensor of indices indicating which rows to select from phase_dist_params.
        :param feature_type: Indicates which feature parameters to use.
        :returns: A tensor with selected distributions based on nodes.
        """
        feature_dist_params = self.phase_dist_params if feature_type == 'phase' else self.new_edge_dist_params if feature_type == 'new_edge' else self.transfer_edge_dist_params
        # Obtain batch indices for each element in nodes
        batch_indices = torch.arange(nodes.size(0)).view(-1, 1).expand_as(nodes)
        # Select the rows from either phase or new edge probabilities
        selected_distributions = feature_dist_params[batch_indices, nodes]
        return selected_distributions

    def _sample_features(self, feature_type: Literal['phase'] | Literal['new_edge'],
                         nodes: torch.Tensor) -> torch.Tensor:
        return self._sample_from_selected_dist_params(self._select_feature_dist_params(nodes, feature_type))

    def _feature_log_probs(self, feature_type: Literal['phase'] | Literal['new_edge'], nodes: torch.Tensor,
                           features: torch.Tensor) -> torch.Tensor:
        selected_feature_dist_params = self._select_feature_dist_params(nodes, feature_type)
        assert_not_all_zero(selected_feature_dist_params)
        batch_size, num_nodes = features.shape
        # Generate a tensor of batch indices to pair with each node index
        batch_indices = torch.arange(batch_size).view(-1, 1).expand_as(features)
        try:
            feature_log_probs = safe_log(selected_feature_dist_params)[batch_indices, torch.arange(num_nodes), features]
            return feature_log_probs
        except IndexError as error:
            # print('batch_indices = ', batch_indices)
            # print('selected_feature_dist_params = ', selected_feature_dist_params)
            # print('features = ', features)
            # print('torch.arange(num_nodes) = ', torch.arange(num_nodes))
            raise error

    def _sample_transfer_edges(self, nodes: torch.Tensor) -> torch.Tensor:
        params = self._select_feature_dist_params(nodes, 'transfer_edge')
        # print('transfer_edge_params = ', params)
        return MultivariateBernoulli(params).sample()

    def _transfer_edge_log_probs(self, nodes: torch.Tensor, transfer_edges: torch.Tensor) -> torch.Tensor:
        # print('transfer_edges = ', transfer_edges)
        # print('self._select_transfer_edge_dist_params = ', self._select_feature_dist_params(nodes, 'transfer_edge'))
        return MultivariateBernoulli(self._select_feature_dist_params(nodes, 'transfer_edge')).log_prob(
            transfer_edges.float())

    def prob(self, sampled_actions: torch.Tensor) -> torch.Tensor:
        return self.log_prob(sampled_actions).exp()

    def log_prob(self, sampled_actions: torch.Tensor) -> torch.Tensor:
        """
        :param sampled_actions: B x K x L tensor of actions. Each action has the form
                                [type, node, phase, new edges, old edges ...], where all entries after node are 0
                                for non-f-right actions. All actions are padded to the same length with zeroes.
                                sampled_actions_batch[b] is the set of actions sampled at some step in a trajectory.
        :return: The log probability of each sampled action.
        """
        action_types = sampled_actions[:, :, 0]
        nodes = sampled_actions[:, :, 1]
        phases = sampled_actions[:, :, 2]
        new_edges = sampled_actions[:, :, 3]
        transfer_edges = sampled_actions[:, :, 4:]
        try:
            action_type_log_probs = self._action_type_log_probs(action_types)
            node_log_probs = self._node_log_probs(action_types, nodes)
            phase_log_probs = self._feature_log_probs('phase', nodes, phases)
            new_edge_log_probs = self._feature_log_probs('new_edge', nodes, new_edges)
            transfer_edge_log_probs = self._transfer_edge_log_probs(nodes, transfer_edges)
            return torch.stack(
                (action_type_log_probs, node_log_probs, phase_log_probs, new_edge_log_probs, transfer_edge_log_probs),
                dim=-1).sum(dim=-1)
        except IndexError as error:
            # print('sampled_actions = ', sampled_actions)
            # print('\n')
            # print('mixture_dist_params.shape = ', self.mixture_dist_params.shape)
            # print('node_dist_params.shape = ', self.node_dist_params.shape)
            # # print('phase_dist_params.shape = ', self.phase_dist_params.shape)
            # # print('new_edge_dist_params.shape = ', self.new_edge_dist_params.shape)
            # # print('transfer_edge_dist_params.shape = ', self.transfer_edge_dist_params.shape)
            # print('\n')
            # print('mixture_dist_params = ', self.mixture_dist_params)
            # print('node_dist_params = ', self.node_dist_params)
            # print('phase_dist_params = ', self.phase_dist_params)
            # print('new_edge_dist_params = ', self.new_edge_dist_params)
            # print('transfer_edge_dist_params = ', self.transfer_edge_dist_params)
            raise error

    def sample(self, k: int) -> torch.Tensor:
        """
        Each of the 'nodes' sampled from the node type distributions (e.g. frz_node_dist_params) is an index into the
        node set for that specific type, where the indices start at zero. For example, if the first two items of an action
        are [1, 3], then the action is to apply the frz rewrite corresponding to the fourth node in the frz node set.

        :param k: The number of samples to produce.
        :return: K x L tensor of actions.
        """
        action_types = self._sample_action_types(k)
        # print('sampled_action_types = ', action_types)
        nodes = self._sample_nodes(action_types)
        # print('sampled_nodes = ', nodes)
        phases = self._sample_features('phase', nodes)
        # print('sampled_phases = ', phases)
        new_edges = self._sample_features('new_edge', nodes)
        # print('sampled_new_edges = ', new_edges)
        transfer_edges = self._sample_transfer_edges(nodes)
        return torch.cat((torch.stack((action_types, nodes, phases, new_edges), dim=-1), transfer_edges), dim=-1).long()

    @staticmethod
    def entropy(device: torch.device = 'cpu') -> torch.Tensor:
        """
        It isn't obvious how to calculate the exact entropy for the entire distribution. We need to take a tractable
        upper bound. Calculating the exact entropy for just the Bernoulli component is also intractable, since the
        support can't easily be enumerated as n grows. So, we need to make two concessions:

        1. Use a mixture of independent multivariate bernoulli distributions instead of a non-independent single
           multi-variate bernoulli mixture.
        2. Optimize an upper bound of the entropy instead of the exact entropy.

        Those concessions, along with the following facts:

        1. H(A, B) <= H(A) + H(B)
        2. H(m1 * A + (1 - m1) * B) <= m1 * H(A) + m2 * H(B)

        will allow us to optimize an upper bound on the distribution.
        """
        return torch.tensor(0.).to(device)
