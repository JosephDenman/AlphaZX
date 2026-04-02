import torch.nn
import torch_geometric as pyg

from alphazx.diagram.match import FRightZMatch, FRightXMatch


class NewEdgeSelector(torch.nn.Module):
    def __init__(self, node_in_channels: int, num_possible_new_edges: int, num_layers: int, dropout: float):
        super(NewEdgeSelector, self).__init__()
        self.num_possible_new_edges = num_possible_new_edges
        self.mlp = pyg.nn.MLP(in_channels=node_in_channels, hidden_channels=node_in_channels,
                              out_channels=num_possible_new_edges, num_layers=num_layers, dropout=dropout, norm='layer_norm')

    def reset_parameters(self):
        self.mlp.reset_parameters()

    def forward(self, x: torch.Tensor, node_types: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        # throw_on_nan(x)
        # Gather node embeddings according to batch
        new_edge_probs = pyg.utils.to_dense_batch(x, batch)[0]
        # Project node embeddings to a vector representing the probabilities of selecting the number of new edges
        new_edge_probs = self.mlp(new_edge_probs).squeeze(dim=-1)
        # Gather node types according to batch
        node_type_batch = pyg.utils.to_dense_batch(node_types, batch, torch.nan)[0]
        # Mask out all non-simple nodes
        node_type_mask = (node_type_batch == FRightZMatch.index) | (node_type_batch == FRightXMatch.index)
        # Create the row to insert for each non-simple node
        replacement_row = torch.zeros(self.num_possible_new_edges, device=new_edge_probs.device)
        replacement_row[0] = 1
        # Insert replacement row for each non-simple node
        new_edge_probs = torch.where(~node_type_mask.unsqueeze(-1).expand_as(new_edge_probs), replacement_row,
                                     new_edge_probs)
        # Softmax over probabilities for each simple node
        new_edge_probs[node_type_mask] = torch.softmax(new_edge_probs[node_type_mask], dim=-1)
        return new_edge_probs
