"""
Neural network evaluation for MCTS.

Provides the bridge between the GameState and the AlphaZXModel:
- Converts a GameState to preprocessed PyG Data
- Runs the model forward pass
- Returns an AlphaZXDistribution (for sampling actions) and a scalar value

The key insight for Sampled MCTS: we don't need to compute a prior probability
for every legal action. Instead, we cache the AlphaZXDistribution at each node
and sample from it on demand during progressive widening. The distribution's
log_prob() method gives us the prior P(a) for PUCT scoring of already-sampled children.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
from torch_geometric.data import Batch

from alphazx.distributions.alpha_zx_dist import AlphaZXDistribution, AlphaZXDistributionParams
from alphazx.models.pre_process import pre_process_single
from alphazx.mcts.game_state import GameState


def _preprocess_data_for_model(data, pe_dim: int):
    """Preprocess a PyG Data object in place: embeddable features + PE + float32 cast.

    This is the shared preprocessing used by both evaluate_state (MCTS) and
    _preprocess_state (replay buffer storage). Factored out so the work is
    done once per state instead of twice.
    """
    # Ensure all float tensors are float32 before preprocessing.
    for key in data.keys():
        attr = data[key]
        if isinstance(attr, torch.Tensor) and attr.is_floating_point():
            data[key] = attr.float()

    prev_dtype = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float32)
        data_out = pre_process_single(data, pe_dim)
    finally:
        torch.set_default_dtype(prev_dtype)

    # Ensure ALL float tensors are float32 after preprocessing.
    for key in data_out.keys():
        attr = data_out[key]
        if isinstance(attr, torch.Tensor) and attr.is_floating_point():
            data_out[key] = attr.float()

    return data_out


@torch.no_grad()
def evaluate_state(
    model: nn.Module,
    state: GameState,
    pe_dim: int,
    device: torch.device = torch.device('cpu'),
) -> tuple[AlphaZXDistribution, float]:
    """
    Run the neural network on a single GameState.

    The model's forward() signature is:
        model(x, edge_index, edge_attr, node_type, batch, pe, graph_ids)

    where `batch` is a per-node tensor of graph indices (all zeros for single graph)
    and `graph_ids` is the diagram's id (a scalar tensor). This matches the calling
    convention in test_zx_game.py.

    :param model: The AlphaZXModel (in eval mode, no_grad context)
    :param state: The GameState to evaluate
    :param pe_dim: Positional encoding dimension
    :param device: Device to run inference on
    :return: (distribution, value) where distribution is an AlphaZXDistribution
             that can be sampled from, and value is a scalar state value estimate
    """
    # Clone data to avoid mutating the original GameState's PyG Data.
    # pre_process_single modifies data.x in place (converting 2D feature vectors
    # to 1D scalar indices), so without cloning, the state would be corrupted.
    data = state.data.clone()
    data = _preprocess_data_for_model(data, pe_dim)

    # Cache the preprocessed data on the state so that self_play's
    # _preprocess_state can reuse it instead of recomputing PE from scratch.
    # This eliminates the second-most-expensive per-step cost (dense PE matmuls).
    # No .clone() needed here: .to(device) below creates new device tensors when
    # device != CPU, and on CPU the model forward pass doesn't mutate inputs.
    state._cached_preprocessed_data = data

    # Move to device (returns self if already on target device)
    data = data.to(device)

    # Construct the batch tensor manually: all nodes belong to graph 0.
    batch_tensor = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

    # Forward pass — graph_ids is data.id (a scalar tensor from zx_match_diagram.to_pyg_data)
    dist_params, value = model(
        data.x, data.edge_index, data.edge_attr,
        data.node_type,
        batch_tensor,
        data.pe,
        data.id,
    )

    distribution = AlphaZXDistribution(dist_params)
    value_scalar = value.item()

    return distribution, value_scalar


@torch.no_grad()
def evaluate_states_batch(
    model: nn.Module,
    states: list[GameState],
    pe_dim: int,
    device: torch.device = torch.device('cpu'),
) -> list[tuple[AlphaZXDistribution, float]]:
    """
    Run the neural network on a batch of GameStates.

    This is more efficient than evaluating states one at a time because
    the GNN forward pass is batched. Use this for batch leaf evaluation
    in parallel MCTS.

    For batched evaluation, we use PyG's Batch which correctly handles
    variable-size graphs by concatenating node/edge features and offsetting
    edge indices. The batch tensor assigns each node to its source graph.

    :param model: The AlphaZXModel
    :param states: List of GameStates to evaluate
    :param pe_dim: Positional encoding dimension
    :param device: Device to run inference on
    :return: List of (distribution, value) tuples, one per state
    """
    if not states:
        return []

    if len(states) == 1:
        # Single state: use the simpler evaluate_state to avoid batch splitting issues
        result = evaluate_state(model, states[0], pe_dim, device)
        return [result]

    # Preprocess each state individually (PE computation is per-graph).
    # Use the shared _preprocess_data_for_model helper which handles float32
    # casting and dtype context management.
    data_list = []
    for s in states:
        d = s.data.clone()
        d = _preprocess_data_for_model(d, pe_dim)
        # Cache preprocessed data on the state so self_play's
        # _preprocess_state can reuse it instead of recomputing PE.
        s._cached_preprocessed_data = d
        data_list.append(d)
    batch = Batch.from_data_list(data_list).to(device)

    # Collect the original graph IDs before batching (as a stacked tensor)
    # Each data.id is a scalar tensor; stack them into a 1D tensor of shape (num_graphs,)
    graph_ids = torch.stack([d.id for d in data_list]).to(device)

    # Forward pass on the full batch
    dist_params, values = model(
        batch.x, batch.edge_index, batch.edge_attr,
        batch.node_type,
        batch.batch,
        batch.pe,
        graph_ids,
    )

    # Split the batch results back into individual distributions.
    # The model passes graph_ids through to AlphaZXDistributionParams unchanged,
    # so dist_params.graph_ids is the stacked tensor we passed in.
    # The selector outputs are batched along dim 0 with one entry per graph.
    results = []
    for i in range(len(states)):
        single_params = AlphaZXDistributionParams(
            graph_ids=dist_params.graph_ids[i:i+1],
            mixture_dist_probs=dist_params.mixture_dist_probs[i:i+1],
            node_dist_probs=dist_params.node_dist_probs[i:i+1],
            phase_dist_probs=dist_params.phase_dist_probs[i:i+1],
            new_edge_dist_probs=dist_params.new_edge_dist_probs[i:i+1],
            transfer_edge_dist_probs=dist_params.transfer_edge_dist_probs[i:i+1],
        )
        distribution = AlphaZXDistribution(single_params)
        value_scalar = values[i].item()
        results.append((distribution, value_scalar))

    return results


def compute_action_prior(
    distribution: AlphaZXDistribution,
    action: tuple,
) -> float:
    """
    Compute the prior probability P(action) from a cached distribution.

    This is used to set the PUCT prior for children that were added via
    progressive widening. The action must have been sampled from this
    distribution (or be a valid action in the same state).

    We compute the joint probability as:
        P(action) = P(type) * P(node|type) * P(phase|node) * P(new_edges|node) * P(transfer_edges|node)

    We compute each component individually rather than using the distribution's
    log_prob() method, which has a graph_ids shape comparison that can fail when
    graph_ids is a scalar tensor (as it is for single-state evaluation).

    :param distribution: The AlphaZXDistribution cached at the parent node
    :param action: Action tuple (graph_id, action_type, node, phase, new_edges, *transfer_edges)
    :return: Prior probability P(action) as a float
    """
    # Unpack action: (graph_id, action_type, node_index, phase, new_edges, *transfer_edges)
    # Create tensors on the same device as the distribution's parameters so that
    # Categorical.log_prob and other ops don't mix MPS/CPU tensors.
    device = distribution.mixture_dist_params.device
    action_type = torch.tensor([action[1]], dtype=torch.long, device=device)
    node = torch.tensor([action[2]], dtype=torch.long, device=device)
    phase = torch.tensor([action[3]], dtype=torch.long, device=device)
    new_edges = torch.tensor([action[4]], dtype=torch.long, device=device)
    transfer_edges = torch.tensor([list(action[5:])], dtype=torch.float, device=device)

    # Compute each component's log probability
    total_log_prob = 0.0

    # P(action_type)
    total_log_prob += distribution.action_type_log_probs(action_type).item()

    # P(node | action_type)
    total_log_prob += distribution.node_log_probs(action_type, node).item()

    # P(phase | node) — for non-F-right actions, phase_dist_params are uniform/zero,
    # but the log_prob still returns a valid value
    total_log_prob += distribution.new_phase_log_probs(node, phase).item()

    # P(new_edges | node)
    total_log_prob += distribution.new_edge_log_probs(node, new_edges).item()

    # P(transfer_edges | node)
    total_log_prob += distribution.transfer_edge_log_probs(node, transfer_edges).item()

    prob = max(1e-8, min(1.0, math.exp(total_log_prob)))
    return prob
