import torch
import torch.nn as nn
import torch_geometric as pyg

from alphazx.diagram.match import FRightZMatch, FRightXMatch


class NewPhaseSelector(nn.Module):
    """Predicts P(phase | node, action_type) for each (action_type, node) pair.

    Conditioning on action_type is implemented via Approach 1: a learned type
    embedding is concatenated to the node embedding before the MLP.  The
    output shape is [B, T, N, P] so that downstream sampling can index
    ``phase_probs[b, t, n]`` to get the phase distribution for node *n*
    when action type *t* was chosen.

    Only F-right match nodes (FRightZMatch, FRightXMatch) receive a learned
    distribution; all other nodes get the deterministic [1, 0, …, 0].
    """

    def __init__(
        self,
        node_in_channels: int,
        num_action_types: int,
        num_possible_phases: int,
        num_layers: int,
        dropout: float,
        type_emb_dim: int = 16,
    ):
        super().__init__()
        self.num_action_types = num_action_types
        self.num_possible_phases = num_possible_phases
        self.type_emb = nn.Embedding(num_action_types, type_emb_dim)
        self.mlp = pyg.nn.MLP(
            in_channels=node_in_channels + type_emb_dim,
            hidden_channels=node_in_channels,
            out_channels=num_possible_phases,
            num_layers=num_layers,
            dropout=dropout,
            norm='layer_norm',
        )

    def reset_parameters(self):
        nn.init.normal_(self.type_emb.weight, std=0.02)
        self.mlp.reset_parameters()

    def forward(
        self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor,
    ) -> torch.Tensor:
        """
        :param x: [num_nodes, C] node embeddings from the policy GPS.
        :param node_types: [num_nodes] integer node types.
        :param batch: [num_nodes] graph membership indices.
        :return: [B, T, N, P] phase probability tensor.
        """
        dense_x, mask = pyg.utils.to_dense_batch(x, batch)       # [B, N, C]
        B, N, C = dense_x.shape
        T = self.num_action_types

        # Build conditioned input: [B, T, N, C + D]
        expanded_x = dense_x.unsqueeze(1).expand(B, T, N, C)     # [B, T, N, C]
        type_embs = self.type_emb.weight                          # [T, D]
        expanded_t = type_embs.unsqueeze(0).unsqueeze(2).expand(B, T, N, -1)
        conditioned = torch.cat([expanded_x, expanded_t], dim=-1) # [B, T, N, C+D]

        # MLP operates on the last dimension; PyG MLP supports arbitrary
        # leading batch dimensions, so this produces [B, T, N, P].
        logits = self.mlp(conditioned)

        # Mask: only F-right match nodes get a learned distribution.
        dense_nt, _ = pyg.utils.to_dense_batch(node_types, batch, fill_value=0)  # [B, N]
        fright_mask = (
            (dense_nt == FRightZMatch.index) | (dense_nt == FRightXMatch.index)
        )                                                         # [B, N]
        # Broadcast to [B, T, N]
        fright_mask = fright_mask.unsqueeze(1).expand(B, T, N)

        # Deterministic replacement row for non-F-right nodes: [1, 0, …, 0]
        replacement = torch.zeros(self.num_possible_phases, device=logits.device)
        replacement[0] = 1.0

        phase_probs = torch.where(
            fright_mask.unsqueeze(-1).expand_as(logits), logits, replacement,
        )
        # Softmax only over positions with a learned distribution
        phase_probs[fright_mask] = torch.softmax(
            phase_probs[fright_mask], dim=-1,
        )

        return phase_probs                                        # [B, T, N, P]
