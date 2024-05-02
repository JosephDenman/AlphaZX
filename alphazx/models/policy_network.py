from typing import Any

import torch
import torch.nn as nn
import torch_geometric as pyg

from alphazx import concatenate_with_neighbor_features
from alphazx.diagram.diagram_generators import clifford_pyg_zx_match_diagram
from alphazx.diagram.match import NODE_METADATA, POSSIBLE_PHASES, METADATA, SIMPLE_NODE_METADATA, BoundaryMatch, \
    SIMPLE_EDGE_METADATA, FRightZMatch, FRightXMatch
from alphazx.distributions.alpha_zx_dist import AZXDistributionParams
from alphazx.models.gps import GPS
from alphazx.models.hgt import HGT
from alphazx.models.pre_process import with_laplacian_pe, with_embeddable_feats

torch.set_printoptions(threshold=10_000)


def aggregate_simple_node_feats(data: pyg.data.HeteroData) -> tuple[pyg.data.HeteroData, pyg.data.HeteroData]:
    simple_subgraph = data.node_type_subgraph(SIMPLE_NODE_METADATA).to_homogeneous(dummy_values=False)
    simple_subgraph.x = concatenate_with_neighbor_features(simple_subgraph.x, simple_subgraph.edge_index)
    simple_subgraph = simple_subgraph.to_heterogeneous()
    del simple_subgraph[BoundaryMatch.abbrev]
    for ntype in SIMPLE_NODE_METADATA:
        del data[ntype]
    for etype in SIMPLE_EDGE_METADATA:
        del data[etype]
    return data, simple_subgraph


class PolicyNetwork(nn.Module):
    def __init__(self,
                 num_node_types: int,
                 num_possible_phases: int,
                 num_possible_new_edges: int,
                 gps_channels: int,
                 gps_node_out_channels: int,
                 gps_edge_in_channels: int,
                 gps_edge_out_channels: int,
                 gps_pe_in_channels: int,
                 gps_pe_out_channels: int,
                 gps_num_layers: int,
                 gps_bias: bool,
                 gps_num_attn_heads: int,
                 gps_attn_type: str,
                 gps_attn_kwargs: dict[str, Any],
                 gps_mlp_hidden_channels: int,
                 hgt_hidden_channels: int,
                 hgt_out_channels: int,
                 hgt_num_heads: int,
                 hgt_num_layers: int,
                 num_pooling_encoder_blocks: int,
                 num_pooling_heads: int,
                 pooling_layer_norm: bool,
                 pooling_dropout: float):
        super(PolicyNetwork, self).__init__()
        self.gps = GPS(num_node_types * num_possible_phases,
                       gps_channels,
                       gps_node_out_channels,
                       gps_edge_in_channels,
                       gps_edge_out_channels,
                       gps_pe_in_channels,
                       gps_pe_out_channels,
                       gps_num_layers,
                       gps_bias,
                       gps_num_attn_heads,
                       gps_attn_type,
                       gps_attn_kwargs,
                       gps_mlp_hidden_channels)
        self.hgt = HGT(METADATA, gps_node_out_channels, hgt_hidden_channels, hgt_out_channels, hgt_num_heads,
                       hgt_num_layers)
        self.z_mlp = pyg.nn.MLP([gps_node_out_channels, 1 + num_possible_phases + num_possible_new_edges])
        self.x_mlp = pyg.nn.MLP([gps_node_out_channels, 1 + num_possible_phases + num_possible_new_edges])
        self.pool_dict = nn.ModuleDict({})
        for ntype in NODE_METADATA:
            if ntype in SIMPLE_NODE_METADATA:
                pool_in_channels = 1 + num_possible_phases + num_possible_new_edges
            else:
                pool_in_channels = hgt_out_channels
            self.pool_dict[ntype] = pyg.nn.GraphMultisetTransformer(pool_in_channels,
                                                                    1,
                                                                    num_pooling_encoder_blocks,
                                                                    num_pooling_heads,
                                                                    pooling_layer_norm,
                                                                    pooling_dropout)

    def _z_nodes(self, data: pyg.data.HeteroData) -> torch.Tensor:
        return self.z_mlp(data['z'].x)

    def forward(self, data: pyg.data.Data) -> AZXDistributionParams:
        print('data = ', data.node_type)
        data.x = self.gps(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)
        # data = data.to_heterogeneous()
        # data = data.set_value_dict('x', self.hgt(data.x_dict, data.edge_index_dict))
        # print('data.post_hgt = ', data)
        data.x[data.node_type == FRightZMatch.index] = self.z_mlp(data.x[data.node_type == FRightZMatch.index])
        data.x[data.node_type == FRightXMatch.index] = self.x_mlp(data.x[data.node_type == FRightXMatch.index])
        print('data.post_mlp = ', data)
        non_basis_node_data, basis_node_data = aggregate_simple_node_feats(data)
        print('data.post_agg = ', data)
        mixture_dist_params = {}
        for ntype in NODE_METADATA:
            mixture_dist_params[ntype] = self.pool_dict[ntype](data[ntype].x, data[ntype].edge_index, data[ntype].batch)
        for ntype in basis_node_data.node_types:
            mixture_dist_params[ntype] = self.pool_dict[ntype](data[ntype].x, data[ntype].edge_index, data[ntype].batch)
        print('mixture_dist_params = ', mixture_dist_params)


def test_policy_network():
    num_node_types = len(NODE_METADATA)
    num_possible_phases = len(POSSIBLE_PHASES)
    num_possible_new_edges = 15
    gps_channels = 6
    gps_node_out_channels = 4
    gps_edge_in_channels = 2
    gps_edge_out_channels = 2
    gps_pe_in_channels = 2
    gps_pe_out_channels = 2
    gps_num_layers = 2
    gps_bias = False
    gps_num_attn_heads = 1
    gps_attn_type = 'multihead'
    gps_attn_kwargs = {}
    gps_mlp_hidden_channels = 5
    hgt_hidden_channels = 8
    hgt_out_channels = 5
    hgt_num_heads = 1
    hgt_num_layers = 2
    num_pooling_encoder_blocks = 2
    num_pooling_heads = 1
    pooling_layer_norm = False
    pooling_dropout = 0.0
    model = PolicyNetwork(num_node_types,
                          num_possible_phases,
                          num_possible_new_edges,
                          gps_channels,
                          gps_node_out_channels,
                          gps_edge_in_channels,
                          gps_edge_out_channels,
                          gps_pe_in_channels,
                          gps_pe_out_channels,
                          gps_num_layers,
                          gps_bias,
                          gps_num_attn_heads,
                          gps_attn_type,
                          gps_attn_kwargs,
                          gps_mlp_hidden_channels,
                          hgt_hidden_channels,
                          hgt_out_channels,
                          hgt_num_heads,
                          hgt_num_layers,
                          num_pooling_encoder_blocks,
                          num_pooling_heads,
                          pooling_layer_norm,
                          pooling_dropout)
    d = clifford_pyg_zx_match_diagram(10, 10)
    d = with_embeddable_feats(d)
    d = with_laplacian_pe(d, 2)
    d.batch = torch.zeros(d.x.size(0), dtype=torch.int64)
    print(model(d))


#
test_policy_network()
