import torch.nn
import torch_geometric as pyg

from alphazx.diagram.match import FRightZMatch, FRightXMatch


class NewPhaseSelector(torch.nn.Module):
    def __init__(self, node_in_channels: int, num_possible_phases: int, num_layers: int, dropout: float):
        super(NewPhaseSelector, self).__init__()
        self.num_possible_phases = num_possible_phases
        self.mlp = pyg.nn.MLP(in_channels=node_in_channels, hidden_channels=node_in_channels,
                              out_channels=num_possible_phases, num_layers=num_layers, dropout=dropout, norm='layer_norm')

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # throw_on_nan(x)
        # Exact same computation as the new edge probs computation
        # Gather node embeddings according to batch
        phase_probs = pyg.utils.to_dense_batch(x, batch)[0]
        # Project node embeddings to a vector representing the probabilities of selecting phases
        phase_probs = self.mlp(phase_probs).squeeze(dim=-1)
        # Gather node types according to batch
        node_type_batch = pyg.utils.to_dense_batch(node_types, batch, torch.nan)[0]
        # Mask out all non-simple nodes
        node_type_mask = (node_type_batch == FRightZMatch.index) | (node_type_batch == FRightXMatch.index)
        # Create the row to insert for each non-simple node
        replacement_row = torch.zeros(self.num_possible_phases, device=phase_probs.device)
        replacement_row[0] = 1
        # Insert replacement row for each non-simple node
        phase_probs = torch.where(~node_type_mask.unsqueeze(-1).expand_as(phase_probs), replacement_row, phase_probs)
        # Softmax over probabilities for each simple node
        phase_probs[node_type_mask] = torch.softmax(phase_probs[node_type_mask], dim=-1)
        return phase_probs
