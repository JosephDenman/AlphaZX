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
    graph_ids: torch.Tensor
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
        :param params: A named-tuple of tensors representing the distribution parameters.

        mixture_dist_params: B x T tensor of mixture (action-type) probabilities.
        node_dist_params: B x T x N tensor of node selection probabilities.
        phase_dist_params: B x T x N x P tensor of phase probabilities,
            conditioned on action type.  For non-F-right (node, type) pairs the
            distribution is deterministic [1, 0, …, 0].
        new_edge_dist_params: B x T x N x E_new, same convention.
        transfer_edge_dist_params: B x T x N x E_trans Bernoulli probabilities,
            conditioned on action type.  Zero for non-F-right (node, type) pairs.
        """
        self.graph_ids = params.graph_ids
        self.mixture_dist_params = params.mixture_dist_probs
        self.B = self.mixture_dist_params.shape[0]
        self.node_dist_params = params.node_dist_probs
        self.phase_dist_params = params.phase_dist_probs
        self.new_edge_dist_params = params.new_edge_dist_probs
        self.transfer_edge_dist_params = params.transfer_edge_dist_probs

    def sample_action_types(self, k: int) -> torch.Tensor:
        return Categorical(probs=self.mixture_dist_params, validate_args=False).sample(torch.Size([k]))

    def action_type_log_probs(self, action_types: torch.Tensor) -> torch.Tensor:
        return Categorical(probs=self.mixture_dist_params, validate_args=False).log_prob(action_types)

    def select_node_dist_params(self, action_types: torch.Tensor) -> torch.Tensor:
        # Action types should already be in the range [0, num_action_types-1] from the policy network
        return self.node_dist_params[torch.arange(self.B), action_types]

    def sample_nodes(self, action_types: torch.Tensor) -> torch.Tensor:
        selected_node_dist_params = self.select_node_dist_params(action_types)
        # Guard against all-zero probability rows which cause NaN after
        # Categorical's internal normalization (0/0) and crash torch.multinomial.
        # This can occur for action types with no valid nodes in the graph.
        selected_node_dist_params = selected_node_dist_params.clamp(min=0)
        row_sums = selected_node_dist_params.sum(dim=-1, keepdim=True)
        selected_node_dist_params = torch.where(
            row_sums > 0,
            selected_node_dist_params,
            torch.ones_like(selected_node_dist_params),
        )
        node_dist = Categorical(probs=selected_node_dist_params, validate_args=False)
        sampled_nodes = node_dist.sample(torch.Size([1]))
        return sampled_nodes

    def node_log_probs(self, action_types: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        selected_node_dist_params = self.select_node_dist_params(action_types)
        node_dist = Categorical(probs=selected_node_dist_params, validate_args=False)
        log_probs = node_dist.log_prob(nodes)
        return log_probs

    def select_feature_dist_params(
        self, feature_type: str, action_types: torch.Tensor, nodes: torch.Tensor,
    ) -> torch.Tensor:
        """Index into [B, T, N, ...] params using (batch, action_type, node)."""
        batch_idx = torch.arange(self.B, device=nodes.device)
        if feature_type == 'phase':
            return self.phase_dist_params[batch_idx, action_types, nodes]
        elif feature_type == 'new_edge':
            return self.new_edge_dist_params[batch_idx, action_types, nodes]
        elif feature_type == 'transfer_edge':
            return self.transfer_edge_dist_params[batch_idx, action_types, nodes]
        else:
            raise Exception('Not implemented')

    def sample_features(
        self, feature_type: str, action_types: torch.Tensor, nodes: torch.Tensor,
    ) -> torch.Tensor:
        if feature_type == 'phase' or feature_type == 'new_edge':
            selected_feature_dist_params = self.select_feature_dist_params(
                feature_type, action_types, nodes,
            )
            # Guard against all-zero rows (see sample_nodes for explanation)
            selected_feature_dist_params = selected_feature_dist_params.clamp(min=0)
            row_sums = selected_feature_dist_params.sum(dim=-1, keepdim=True)
            selected_feature_dist_params = torch.where(
                row_sums > 0,
                selected_feature_dist_params,
                torch.ones_like(selected_feature_dist_params),
            )
            feature_dist = Categorical(probs=selected_feature_dist_params, validate_args=False)
            sampled_feature = feature_dist.sample(torch.Size([1]))
            return sampled_feature
        else:
            raise Exception('Not implemented')

    def feature_log_probs(
        self,
        feature_type: str,
        action_types: torch.Tensor,
        nodes: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        if feature_type == 'phase' or feature_type == 'new_edge':
            selected_feature_dist_params = self.select_feature_dist_params(
                feature_type, action_types, nodes,
            )
            feature_dist = Categorical(probs=selected_feature_dist_params, validate_args=False)
            log_probs = feature_dist.log_prob(features)
            return log_probs
        else:
            raise Exception('Not implemented')

    def sample_phases(self, action_types: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        return self.sample_features('phase', action_types, nodes)

    def new_phase_log_probs(
        self, action_types: torch.Tensor, nodes: torch.Tensor, phases: torch.Tensor,
    ) -> torch.Tensor:
        return self.feature_log_probs('phase', action_types, nodes, phases)

    def sample_new_edges(self, action_types: torch.Tensor, nodes: torch.Tensor) -> torch.Tensor:
        return self.sample_features('new_edge', action_types, nodes)

    def new_edge_log_probs(
        self, action_types: torch.Tensor, nodes: torch.Tensor, new_edges: torch.Tensor,
    ) -> torch.Tensor:
        return self.feature_log_probs('new_edge', action_types, nodes, new_edges)

    def sample_transfer_edges(
        self, action_types: torch.Tensor, nodes: torch.Tensor,
    ) -> torch.Tensor:
        selected_feature_dist_params = self.select_feature_dist_params(
            'transfer_edge', action_types, nodes,
        )
        return MultivariateBernoulli(selected_feature_dist_params).sample(torch.Size([1]))

    def transfer_edge_log_probs(
        self, action_types: torch.Tensor, nodes: torch.Tensor, transfer_edges: torch.Tensor,
    ) -> torch.Tensor:
        selected_feature_dist_params = self.select_feature_dist_params(
            'transfer_edge', action_types, nodes,
        )
        # The stored action may have a different number of transfer edges than
        # the model's output (E_trans depends on max degree in the graph at the
        # time of the forward pass vs. the graph at the time of play).  Align
        # by padding the shorter side with zeros.
        E_model = selected_feature_dist_params.shape[-1]
        E_action = transfer_edges.shape[-1]
        if E_action < E_model:
            pad = torch.zeros(
                *transfer_edges.shape[:-1], E_model - E_action,
                dtype=transfer_edges.dtype, device=transfer_edges.device,
            )
            transfer_edges = torch.cat([transfer_edges, pad], dim=-1)
        elif E_model < E_action:
            pad = torch.zeros(
                *selected_feature_dist_params.shape[:-1], E_action - E_model,
                dtype=selected_feature_dist_params.dtype,
                device=selected_feature_dist_params.device,
            )
            selected_feature_dist_params = torch.cat(
                [selected_feature_dist_params, pad], dim=-1,
            )
        return MultivariateBernoulli(selected_feature_dist_params).log_prob(transfer_edges.float())

    def probs(self, sampled_actions: torch.Tensor) -> torch.Tensor:
        return self.log_prob(sampled_actions).exp()

    def log_prob(self, sampled_actions: torch.Tensor) -> torch.Tensor:
        """
        :param sampled_actions: B x K x L tensor of actions. Each action has the form
                                [graph_id, type, node, phase, new edges, old edges ...], where all entries after node
                                are 0 for non-f-right actions. All actions are padded to the same length with zeroes.
                                sampled_actions_batch[b] is the set of actions sampled at some step in a trajectory.
        :return: (B, K) tensor of log probabilities for each sampled action.
        """
        B, K, L = sampled_actions.shape
        graph_ids = sampled_actions[:, :, 0]
        action_types = sampled_actions[:, :, 1]
        nodes = sampled_actions[:, :, 2]
        phases = sampled_actions[:, :, 3]
        new_edges = sampled_actions[:, :, 4]
        transfer_edges = sampled_actions[:, :, 5:]
        expected_ids = self.graph_ids.long().flatten()  # (B,) even if scalar
        received_ids = graph_ids.long()[:, 0]  # first sample per batch entry → (B,)
        if not torch.equal(received_ids, expected_ids):
            raise Exception(f'Expected graph ids {expected_ids}, received {received_ids}')

        # Flatten (B, K) → (B*K,) so component methods (which expect a [B] batch dim) work.
        # We temporarily expand the distribution params to repeat each batch entry K times.
        flat_action_types = action_types.reshape(B * K)
        flat_nodes = nodes.reshape(B * K)
        flat_phases = phases.reshape(B * K)
        flat_new_edges = new_edges.reshape(B * K)
        flat_transfer_edges = transfer_edges.reshape(B * K, -1)

        # Save original state and expand params for B*K batch
        orig_B = self.B
        orig_mixture = self.mixture_dist_params
        orig_node = self.node_dist_params
        orig_phase = self.phase_dist_params
        orig_new_edge = self.new_edge_dist_params
        orig_transfer_edge = self.transfer_edge_dist_params

        # Repeat each batch entry K times: [B, ...] → [B, 1, ...] → [B, K, ...] → [B*K, ...]
        self.B = B * K
        self.mixture_dist_params = orig_mixture.unsqueeze(1).expand(-1, K, -1).reshape(B * K, -1)
        self.node_dist_params = orig_node.unsqueeze(1).expand(-1, K, *orig_node.shape[1:]).reshape(B * K, *orig_node.shape[1:])
        self.phase_dist_params = orig_phase.unsqueeze(1).expand(-1, K, *orig_phase.shape[1:]).reshape(B * K, *orig_phase.shape[1:])
        self.new_edge_dist_params = orig_new_edge.unsqueeze(1).expand(-1, K, *orig_new_edge.shape[1:]).reshape(B * K, *orig_new_edge.shape[1:])
        self.transfer_edge_dist_params = orig_transfer_edge.unsqueeze(1).expand(-1, K, *orig_transfer_edge.shape[1:]).reshape(B * K, *orig_transfer_edge.shape[1:])

        try:
            action_type_log_probs = self.action_type_log_probs(flat_action_types)
            node_log_probs = self.node_log_probs(flat_action_types, flat_nodes)
            phase_log_probs = self.new_phase_log_probs(flat_action_types, flat_nodes, flat_phases)
            new_edge_log_probs = self.new_edge_log_probs(flat_action_types, flat_nodes, flat_new_edges)
            transfer_edge_log_probs = self.transfer_edge_log_probs(flat_action_types, flat_nodes, flat_transfer_edges)
        finally:
            # Restore original state
            self.B = orig_B
            self.mixture_dist_params = orig_mixture
            self.node_dist_params = orig_node
            self.phase_dist_params = orig_phase
            self.new_edge_dist_params = orig_new_edge
            self.transfer_edge_dist_params = orig_transfer_edge

        stacked_probs = torch.stack(
            (action_type_log_probs, node_log_probs, phase_log_probs, new_edge_log_probs, transfer_edge_log_probs),
            dim=-1)
        # Reshape from (B*K,) back to (B, K)
        return stacked_probs.sum(dim=-1).reshape(B, K)

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
        phases = self.sample_phases(action_types, nodes)[0]
        new_edges = self.sample_new_edges(action_types, nodes)[0]
        transfer_edges = self.sample_transfer_edges(action_types, nodes)[0]
        stacked = torch.stack((action_types, nodes, phases, new_edges), dim=-1)  # (K, B, 4)
        # Categorical.sample returns (sample_shape, *batch_shape) = (K, B, ...).
        # Transpose to batch-first (B, K, ...) to match log_prob's expected layout.
        stacked = stacked.transpose(0, 1)  # (B, K, 4)
        transfer_edges = transfer_edges.transpose(0, 1)  # (B, K, E_trans)
        B, K, _ = stacked.shape
        gids = self.graph_ids.reshape(-1, 1, 1).expand(B, K, 1)  # (B, K, 1)
        samples = torch.cat((gids, stacked, transfer_edges), dim=-1).long()
        return samples

    def entropy(self) -> torch.Tensor:
        """
        Calculate an approximation of the distribution entropy for exploration.
        This uses the upper bound: H(mixture) + sum(H(components | mixture))
        """
        # Entropy of the mixture distribution (action type selection)
        mixture_entropy = Categorical(probs=self.mixture_dist_params, validate_args=False).entropy()

        # Average entropy of node selection across action types
        node_entropies = []
        for action_type in range(self.mixture_dist_params.shape[1]):
            action_type_tensor = torch.full((self.B,), action_type, device=self.mixture_dist_params.device)
            try:
                node_params = self.select_node_dist_params(action_type_tensor)
                # Only compute entropy if there are valid nodes for this action type
                if not torch.all(node_params == 0):
                    node_entropy = Categorical(probs=node_params + 1e-8, validate_args=False).entropy()
                    node_entropies.append(node_entropy)
            except Exception:
                # Skip if no valid nodes for this action type
                continue

        if node_entropies:
            avg_node_entropy = torch.stack(node_entropies).mean()
        else:
            avg_node_entropy = torch.tensor(0.0, device=self.mixture_dist_params.device)

        # Combine entropies (this is an upper bound)
        total_entropy = mixture_entropy + avg_node_entropy * 0.5  # Weight node entropy less

        return total_entropy.mean()  # Average over batch
