from typing import Literal

import torch
from torch.distributions.categorical import Categorical

from models.bernoulli_mixture import MultivariateBernoulliMixture


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

    def __init__(self,
                 mixture_dist_params: torch.Tensor,
                 frz_node_dist_params: torch.Tensor,
                 flz_node_dist_params: torch.Tensor,
                 phase_dist_params: torch.Tensor,
                 new_edges_dist_params: torch.Tensor,
                 transfer_edges_dist_params: torch.Tensor):
        """
        :param mixture_dist_params: B x T tensor of mixture probabilities. mixture_dist_params[b] is the mixture probabilities
                                    at some step in a trajectory. It is assumed to be a softmax output of a DNN. When B = 1,
                                    we are in the MCTS portion of the algorithm.
        :param frz_node_dist_params: B x N tensor of frz-node selection probabilities. For a node n in batch b that does
                                     not represent a frz-node, frz_node_dist_params[b, n] = 0.
        :param frz_node_dist_params: B x N tensor of flz-node selection probabilities. For a node n in batch b that does
                                     not represent a flz-node, flz_node_dist_params[b, n] = 0.
        :param phase_dist_params: B x N x P tensor of phase probabilities. For a node n in batch b that does not
                                  represent an f-right match, phase_dist_params[b, n] = torch.zeroes((P, ))
        :param new_edges_dist_params: B x N x E_new tensor of new edge probabilities. For a node n in batch b that does not
                                      represent an f-right match, new_edges_dist_params[b, n] is all zeros.
        :param transfer_edges_dist_params: B x N x E_trans x (E_trans + 1) tensor of probabilities for each node. For a node n in batch b
                                           that does not represent an f-right match, transfer_edges_dist_params[b, n] has a single 1 entry
                                           in the top left corner of the innermost 2D tensor. All other entries in the innermost 2D tensor are 0.
                                           Samples drawn from this distribution are all zeros (no edges are selected to be transferred).
        """
        self.mixture_dist_params = mixture_dist_params
        self.frz_node_dist_params = frz_node_dist_params
        self.flz_node_dist_params = flz_node_dist_params
        self.phase_dist_params = phase_dist_params
        self.new_edges_dist_params = new_edges_dist_params
        self.transfer_edges_dist_params = transfer_edges_dist_params

    def _sample_action_types(self, k: int) -> torch.Tensor:
        return Categorical(probs=self.mixture_dist_params).sample(torch.Size([k])).T

    def _action_type_log_probs(self, action_types: torch.Tensor) -> torch.Tensor:
        return torch.gather(self.mixture_dist_params.log(), 1, action_types)

    def _select_node_dist_params(self, action_types: torch.Tensor) -> torch.Tensor:
        # Reshape action_types for broadcasting over the distributions
        action_types_expanded = action_types.unsqueeze(-1).expand(-1, -1, self.frz_node_dist_params.size(-1))
        selected_node_dist_params = torch.where(action_types_expanded == 0,
                                                self.frz_node_dist_params.unsqueeze(1).repeat(1, action_types.size(1),
                                                                                              1),
                                                self.flz_node_dist_params.unsqueeze(1).repeat(1, action_types.size(1),
                                                                                              1))
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
        selected_node_dist_params = self._select_node_dist_params(action_types)
        batch_size, num_nodes = nodes.shape
        # Generate a tensor of batch indices to pair with each node index
        batch_indices = torch.arange(batch_size).view(-1, 1).expand_as(nodes)
        # Select the distribution parameters for each node
        node_log_probs = selected_node_dist_params.log()[batch_indices, torch.arange(num_nodes), nodes]
        return node_log_probs

    def _select_feature_dist_params(self, nodes: torch.Tensor,
                                    feature_type: Literal['phase'] | Literal['new_edge'] | Literal[
                                        'transfer_edge']) -> torch.Tensor:
        """
        Selects rows from feature distribution parameters based on indices in nodes.

        :param nodes: A tensor of indices indicating which rows to select from phase_dist_params.
        :param feature_type: Indicates which feature parameters to use.
        :returns: A tensor with selected distributions based on nodes.
        """
        feature_dist_params = self.phase_dist_params if feature_type == 'phase' else self.new_edges_dist_params if feature_type == 'new_edge' else self.transfer_edges_dist_params
        # Obtain batch indices for each element in nodes
        batch_indices = torch.arange(nodes.size(0)).view(-1, 1).expand_as(nodes)
        # Select the rows from phase_dist_params
        selected_distributions = feature_dist_params[batch_indices, nodes]
        return selected_distributions

    def _sample_features(self, feature_type: Literal['phase'] | Literal['new_edge'],
                         nodes: torch.Tensor) -> torch.Tensor:
        return self._sample_from_selected_dist_params(self._select_feature_dist_params(nodes, feature_type))

    def _feature_log_probs(self, feature_type: Literal['phase'] | Literal['new_edge'], nodes: torch.Tensor,
                           features: torch.Tensor) -> torch.Tensor:
        selected_feature_dist_params = self._select_feature_dist_params(nodes, feature_type)
        batch_size, num_nodes = features.shape
        # Generate a tensor of batch indices to pair with each node index
        batch_indices = torch.arange(batch_size).view(-1, 1).expand_as(features)
        # Select the corresponding distribution parameters for each node
        feature_log_probs = selected_feature_dist_params.log()[batch_indices, torch.arange(num_nodes), features]
        return feature_log_probs

    def _sample_transfer_edges(self, nodes: torch.Tensor) -> torch.Tensor:
        return MultivariateBernoulliMixture(self._select_feature_dist_params(nodes, 'transfer_edge')).sample()

    def _transfer_edge_log_probs(self, nodes: torch.Tensor, transfer_edges: torch.Tensor) -> torch.Tensor:
        return MultivariateBernoulliMixture(self._select_feature_dist_params(nodes, 'transfer_edge')).log_prob(
            transfer_edges.float())

    def log_prob(self, sampled_actions: torch.Tensor) -> torch.Tensor:
        """
        :param sampled_actions: B x K x L tensor of actions. Each action has the form
                                [type, node, phase, new edges, old edges ...], where all entries after node are 0
                                for non-f-right actions. All actions are padded to the same length with zeroes.
                                sampled_actions_batch[b] is the set of actions sampled at some step in a trajectory.
        :return: The log probability of each sampled action.
        """
        actions = sampled_actions[:, :, 0]
        nodes = sampled_actions[:, :, 1]
        phases = sampled_actions[:, :, 2]
        new_edges = sampled_actions[:, :, 3]
        transfer_edges = sampled_actions[:, :, 4:]
        action_type_log_probs = self._action_type_log_probs(actions)
        node_log_probs = self._node_log_probs(actions, nodes)
        phase_log_probs = self._feature_log_probs('phase', nodes, phases)
        new_edge_log_probs = self._feature_log_probs('new_edge', nodes, new_edges)
        transfer_edge_log_probs = self._transfer_edge_log_probs(nodes, transfer_edges)
        return torch.stack(
            (action_type_log_probs, node_log_probs, phase_log_probs, new_edge_log_probs, transfer_edge_log_probs),
            dim=-1).sum(dim=-1)

    def sample(self, k: int) -> torch.Tensor:
        """
        :param k: The number of samples to produce.
        :return: K x L tensor of actions.
        """
        action_types = self._sample_action_types(k)
        nodes = self._sample_nodes(action_types)
        phases = self._sample_features('phase', nodes)
        new_edges = self._sample_features('new_edge', nodes)
        transfer_edges = self._sample_transfer_edges(nodes)
        return torch.cat((torch.stack((action_types, nodes, phases, new_edges), dim=-1), transfer_edges), dim=-1).long()
