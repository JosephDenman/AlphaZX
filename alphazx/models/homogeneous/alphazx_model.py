from typing import Any

import torch
import torch.nn as nn

from alphazx.diagram import METADATA, POSSIBLE_PHASES
from alphazx.distributions.alpha_zx_dist import AlphaZXDistributionParams
from alphazx.models.homogeneous.prediction_network import PredictionNetwork
from alphazx.models.homogeneous.representation_network import RepresentationNetwork


class AlphaZXModel(nn.Module):
    def __init__(self,
                 num_node_types: int,
                 num_possible_phases: int,
                 num_possible_new_edges: int,
                 node_embedding_channels: int,
                 num_edge_embeddings: int,
                 edge_embedding_channels: int,
                 pe_in_channels: int,
                 pe_out_channels: int):
        super(AlphaZXModel, self).__init__()
        self.representation_network = RepresentationNetwork(num_node_types,
                                                            num_possible_phases,
                                                            node_embedding_channels,
                                                            node_embedding_channels,
                                                            num_edge_embeddings,
                                                            edge_embedding_channels,
                                                            pe_in_channels,
                                                            pe_out_channels)
        self.prediction_network = PredictionNetwork(num_node_types,
                                                    num_possible_phases,
                                                    num_possible_new_edges,
                                                    node_embedding_channels,
                                                    edge_embedding_channels)

    def reset_parameters(self):
        self.representation_network.reset_parameters()
        self.prediction_network.reset_parameters()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor, node_type: torch.Tensor,
                batch: torch.Tensor, pe: torch.Tensor, graph_ids: torch.Tensor,
                edge_type: torch.Tensor | None = None) -> tuple[AlphaZXDistributionParams, torch.Tensor]:
        # edge_type is accepted for interface compatibility with AlphaZXHeteroModel
        # but is unused by the homogeneous model (GPS does not use edge types).
        x, edge_attr = self.representation_network(x, edge_index, edge_attr, batch, pe)
        policy, value = self.prediction_network(x, edge_index, edge_attr, node_type, batch, graph_ids)
        return policy, value
