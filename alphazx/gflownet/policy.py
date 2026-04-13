"""
Forward and backward policy networks for GFlowNet training.

The forward policy P_F reuses the existing AlphaZXModel backbone (GNN →
representation → policy head).  The model's AlphaZXDistribution provides
the per-component distributions that the sampler queries individually:

    P(type), P(node|type), P(phase|type,node), P(edge|type,node), P(transfer|type,node)

Each of these becomes a separate sub-action in the GFlowNet DAG, so
the policy no longer needs to compute a joint log-prob or handle
per-head normalisation — that's taken care of structurally.

For Trajectory Balance, we also maintain a learnable log Z parameter
(the log partition function).
"""

from __future__ import annotations

import logging
import math

import torch
import torch.nn as nn

from alphazx.distributions.alpha_zx_dist import AlphaZXDistribution
from alphazx.shared.evaluate import _preprocess_data_for_model
from alphazx.shared.game_state import GameState

logger = logging.getLogger(__name__)


class GFlowNetForwardPolicy(nn.Module):
    """Forward policy P_F for GFlowNet trajectory sampling.

    Wraps an existing AlphaZXModel (or AlphaZXHeteroModel).  The model's
    policy head outputs an AlphaZXDistribution whose individual component
    distributions are queried by the sampler for each sub-action.

    The value head is repurposed as the state-flow estimator log F(s),
    used by Detailed Balance and Flow Matching losses.  For Trajectory
    Balance, a separate learnable log Z is used instead.
    """

    def __init__(self, model: nn.Module, pe_dim: int = 20):
        super().__init__()
        self.model = model
        self.pe_dim = pe_dim
        # Learnable log partition function for TB loss
        self.log_Z = nn.Parameter(torch.tensor(0.0))

    def forward(
        self, state: GameState, device: str = 'cpu',
    ) -> tuple[AlphaZXDistribution, torch.Tensor]:
        """Run the model on a state, returning (distribution, state_flow).

        Does NOT use @torch.no_grad() so gradients flow through for TB loss.
        The caller is responsible for wrapping in torch.no_grad() when
        gradients are not needed (e.g. evaluation / data collection).

        The distribution's per-component methods (action_type_log_probs,
        node_log_probs, new_phase_log_probs, etc.) are called directly by
        the sampler for each sub-action step.
        """
        data = state.data.clone()
        data = _preprocess_data_for_model(data, self.pe_dim)
        data = data.to(device)

        batch_tensor = torch.zeros(data.x.shape[0], dtype=torch.long, device=device)

        dist_params, value = self.model(
            data.x, data.edge_index, data.edge_attr,
            data.node_type,
            batch_tensor,
            data.pe,
            data.id,
            edge_type=getattr(data, 'edge_type', None),
        )

        distribution = AlphaZXDistribution(dist_params)
        return distribution, value


class UniformBackwardPolicy:
    """Uniform backward policy P_B for Trajectory Balance.

    In the decomposed design, each sub-action has its own backward
    probability.  For a uniform backward policy:

      - Choose type:     P_B = 1 / num_available_types
      - Choose node:     P_B = 1 / num_nodes_of_this_type
      - Choose phase:    P_B = 1 / num_possible_phases
      - Choose new_edge: P_B = 1 / num_possible_new_edges
      - Choose transfer: P_B = 1 / 2^num_bits  (each bit independent)
    """

    @staticmethod
    def log_prob(num_choices: int) -> float:
        """log P_B = -log(num_choices)."""
        if num_choices <= 0:
            return 0.0
        return -math.log(num_choices)
