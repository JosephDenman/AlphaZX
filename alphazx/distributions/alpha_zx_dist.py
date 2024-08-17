from typing import NamedTuple

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


def check_non_zero_rows(t: torch.Tensor) -> None:
    if torch.any(torch.all((t == 0.), dim=-1)).item():
        raise Exception(f'Input tensor {t} contains all zero rows')


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

    def __init__(self, params: AlphaZXDistributionParams):
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
        check_non_zero_rows(params.mixture_dist_probs)
        self.mixture_dist_params = params.mixture_dist_probs
        self.B = self.mixture_dist_params.shape[0]
        self.node_dist_params = params.node_dist_probs
        check_non_zero_rows(params.phase_dist_probs)
        self.phase_dist_params = params.phase_dist_probs
        check_non_zero_rows(params.new_edge_dist_probs)
        self.new_edge_dist_params = params.new_edge_dist_probs
        self.transfer_edge_dist_params = params.transfer_edge_dist_probs

    def sample_action_types(self, k: int) -> torch.Tensor:
        return Categorical(probs=self.mixture_dist_params, validate_args=True).sample(torch.Size([k]))

    def action_type_log_probs(self, action_types: torch.Tensor) -> torch.Tensor:
        return Categorical(probs=self.mixture_dist_params, validate_args=True).log_prob(action_types)

    def select_node_dist_params(self, action_types: torch.Tensor) -> torch.Tensor:
        selected_node_dist_params = self.node_dist_params[torch.arange(self.B), action_types - 11]
        try:
            check_non_zero_rows(selected_node_dist_params)
        except Exception as error:
            print('action_types = ', action_types)
            print('mixture_dist_params = ', self.mixture_dist_params)
            print('node_dist_params = ', self.node_dist_params)
            raise error
        return selected_node_dist_params

    def sample_nodes(self, action_types: torch.Tensor) -> torch.Tensor:
        selected_node_dist_params = self.select_node_dist_params(action_types)
        check_non_zero_rows(selected_node_dist_params)
        node_dist = Categorical(probs=selected_node_dist_params, validate_args=True)
        sampled_nodes = node_dist.sample(torch.Size([1]))
        return sampled_nodes

    def node_log_probs(self, action_types: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        selected_node_dist_params = self.select_node_dist_params(action_types)
        assert_not_all_zero(selected_node_dist_params)
        node_dist = Categorical(probs=selected_node_dist_params, validate_args=True)
        log_probs = node_dist.log_prob(nodes)
        return log_probs

    def select_feature_dist_params(self, feature_type: str, nodes: torch.Tensor) -> torch.Tensor:
        if feature_type == 'phase':
            return self.phase_dist_params[torch.arange(self.B), nodes]
        elif feature_type == 'new_edge':
            return self.new_edge_dist_params[torch.arange(self.B), nodes]
        elif feature_type == 'transfer_edge':
            return self.transfer_edge_dist_params[torch.arange(self.B), nodes]
        else:
            raise Exception('Not implemented')

    def sample_features(self, feature_type: str, nodes: torch.Tensor) -> torch.Tensor:
        if feature_type == 'phase' or feature_type == 'new_edge':
            selected_feature_dist_params = self.select_feature_dist_params(feature_type, nodes)
            check_non_zero_rows(selected_feature_dist_params)
            feature_dist = Categorical(probs=selected_feature_dist_params, validate_args=True)
            sampled_feature = feature_dist.sample(torch.Size([1]))
            return sampled_feature
        else:
            raise Exception('Not implemented')

    def feature_log_probs(self, feature_type: str, nodes: torch.Tensor, features: torch.Tensor) -> torch.Tensor:
        if feature_type == 'phase' or feature_type == 'new_edge':
            selected_feature_dist_params = self.select_feature_dist_params(feature_type, nodes)
            assert_not_all_zero(selected_feature_dist_params)
            feature_dist = Categorical(probs=selected_feature_dist_params, validate_args=True)
            log_probs = feature_dist.log_prob(features)
            return log_probs
        else:
            raise Exception('Not implemented')

    def sample_phases(self, nodes: torch.Tensor) -> torch.Tensor:
        return self.sample_features('phase', nodes)

    def new_phase_log_probs(self, nodes: torch.Tensor, phases: torch.Tensor) -> torch.Tensor:
        return self.feature_log_probs('phase', nodes, phases)

    def sample_new_edges(self, nodes: torch.Tensor) -> torch.Tensor:
        return self.sample_features('new_edge', nodes)

    def new_edge_log_probs(self, nodes: torch.Tensor, new_edges: torch.Tensor) -> torch.Tensor:
        return self.feature_log_probs('new_edge', nodes, new_edges)

    def sample_transfer_edges(self, nodes: torch.Tensor) -> torch.Tensor:
        selected_feature_dist_params = self.select_feature_dist_params('transfer_edge', nodes)
        return MultivariateBernoulli(selected_feature_dist_params).sample(torch.Size([1]))

    def transfer_edge_log_probs(self, nodes: torch.Tensor, transfer_edges: torch.Tensor) -> torch.Tensor:
        selected_feature_dist_params = self.select_feature_dist_params('transfer_edge', nodes)
        return MultivariateBernoulli(selected_feature_dist_params).log_prob(transfer_edges.float())

    def probs(self, sampled_actions: torch.Tensor) -> torch.Tensor:
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
        action_type_log_probs = self.action_type_log_probs(action_types)
        node_log_probs = self.node_log_probs(action_types, nodes)
        phase_log_probs = self.new_phase_log_probs(nodes, phases)
        new_edge_log_probs = self.new_edge_log_probs(nodes, new_edges)
        transfer_edge_log_probs = self.transfer_edge_log_probs(nodes, transfer_edges)
        stacked_probs = torch.stack(
            (action_type_log_probs, node_log_probs, phase_log_probs, new_edge_log_probs, transfer_edge_log_probs),
            dim=-1)
        return stacked_probs.sum(dim=-1)

    def sample(self, k: int) -> torch.Tensor:
        """
        Each of the 'nodes' sampled from the node type distributions (e.g. frz_node_dist_params) is an index into the
        node set for that specific type, where the indices start at zero. For example, if the first two items of an action
        are [1, 3], then the action is to apply the frz rewrite corresponding to the fourth node in the frz node set.

        :param k: The number of samples to produce.
        :return: K x L tensor of actions.
        """
        action_types = self.sample_action_types(k)
        nodes = self.sample_nodes(action_types)[0]
        phases = self.sample_phases(nodes)[0]
        new_edges = self.sample_new_edges(nodes)[0]
        transfer_edges = self.sample_transfer_edges(nodes)[0]
        return torch.cat((torch.stack((action_types, nodes, phases, new_edges), dim=-1), transfer_edges), dim=-1).long()

    @staticmethod
    def entropy() -> torch.Tensor:
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
        return torch.tensor(0.)
