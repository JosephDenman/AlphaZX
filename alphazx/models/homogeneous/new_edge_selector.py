import torch
import torch.nn as nn
import torch_geometric as pyg

from alphazx.diagram.match import FRightZMatch, FRightXMatch


class NewEdgeSelector(nn.Module):
    """Predicts P(new_edges | node, action_type) for each (action_type, node) pair.

    Same conditioning strategy as :class:`NewPhaseSelector`: a learned type
    embedding is concatenated to the node embedding before the MLP.  Output
    shape is [B, T, N, E_new].
    """

    def __init__(
        self,
        node_in_channels: int,
        num_action_types: int,
        num_possible_new_edges: int,
        num_layers: int,
        dropout: float,
        type_emb_dim: int = 16,
    ):
        super().__init__()
        self.num_action_types = num_action_types
        self.num_possible_new_edges = num_possible_new_edges
        self.type_emb = nn.Embedding(num_action_types, type_emb_dim)
        self.mlp = pyg.nn.MLP(
            in_channels=node_in_channels + type_emb_dim,
            hidden_channels=node_in_channels,
            out_channels=num_possible_new_edges,
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
        :return: [B, T, N, E_new] new-edge probability tensor.
        """
        dense_x, mask = pyg.utils.to_dense_batch(x, batch)       # [B, N, C]
        B, N, C = dense_x.shape
        T = self.num_action_types

        # Conditioned input: [B, T, N, C + D]
        expanded_x = dense_x.unsqueeze(1).expand(B, T, N, C)
        type_embs = self.type_emb.weight                          # [T, D]
        expanded_t = type_embs.unsqueeze(0).unsqueeze(2).expand(B, T, N, -1)
        conditioned = torch.cat([expanded_x, expanded_t], dim=-1)

        logits = self.mlp(conditioned)                            # [B, T, N, E_new]

        # Mask: only F-right match nodes get a learned distribution
        dense_nt, _ = pyg.utils.to_dense_batch(node_types, batch, fill_value=0)
        fright_mask = (
            (dense_nt == FRightZMatch.index) | (dense_nt == FRightXMatch.index)
        ).unsqueeze(1).expand(B, T, N)                            # [B, T, N]

        replacement = torch.zeros(self.num_possible_new_edges, device=logits.device)
        replacement[0] = 1.0

        edge_probs = torch.where(
            fright_mask.unsqueeze(-1).expand_as(logits), logits, replacement,
        )
        edge_probs[fright_mask] = torch.softmax(
            edge_probs[fright_mask], dim=-1,
        )

        return edge_probs                                         # [B, T, N, E_new]
