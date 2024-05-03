from typing import Any

import torch
import torch.nn as nn
import torch_geometric as pyg

from alphazx import concatenate_with_neighbor_features
from alphazx.diagram.diagram_generators import clifford_pyg_zx_match_diagram
from alphazx.diagram.match import NODE_METADATA, POSSIBLE_PHASES, METADATA, SIMPLE_NODE_METADATA, BoundaryMatch, \
    SIMPLE_EDGE_METADATA
from alphazx.distributions.alpha_zx_dist import AZXDistributionParams
from alphazx.models.gps import GPS
from alphazx.models.hgt import HGT
from alphazx.models.pre_process import with_laplacian_pe, with_embeddable_feats

torch.set_printoptions(threshold=100_000)


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
        self.mlp = pyg.nn.MLP([gps_node_out_channels, 1 + num_possible_phases + num_possible_new_edges])
        # self.trans_enc_dict = nn.ModuleDict({})
        # Use on the result of neighbor concatenation
        # for ntype in NODE_METADATA:
        #     self.trans_enc_dict[ntype] = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=512, nhead=8))
        # self.pool_dict = nn.ModuleDict({})
        # for ntype in NODE_METADATA:
        #     self.pool_dict[ntype] = pyg.nn.GraphMultisetTransformer(1 + num_possible_phases + num_possible_new_edges,
        #                                                             1,
        #                                                             num_pooling_encoder_blocks,
        #                                                             num_pooling_heads,
        #                                                             pooling_layer_norm,
        #                                                             pooling_dropout)
        self.pool = pyg.nn.GraphMultisetTransformer(1 + num_possible_phases + num_possible_new_edges,
                                                    1,
                                                    num_pooling_encoder_blocks,
                                                    num_pooling_heads,
                                                    pooling_layer_norm,
                                                    pooling_dropout)

    def _z_nodes(self, data: pyg.data.HeteroData) -> torch.Tensor:
        return self.z_mlp(data['z'].x)

    def forward(self, data: pyg.data.Data) -> AZXDistributionParams:
        # TODO: There can be
        print('data = ', data)
        data.x = self.gps(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)
        data.x = self.mlp(data.x)
        print('data.x = ', data.x)
        # data.x = concatenate_with_neighbor_features(data.x, data.edge_index)
        # flattened_feats = torch.flatten(data.x, -2)
        # print('flattened_feats = ', flattened_feats)
        mixture_dist_params = self.pool(data.x, data.node_type)
        print('data.node_type = ', data.node_type)
        print('mixture_dist_params = ', mixture_dist_params)


def policy_network():
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
    d = clifford_pyg_zx_match_diagram(100, 100)
    d = with_embeddable_feats(d)
    d = with_laplacian_pe(d, 2)
    d.batch = torch.zeros(d.x.size(0), dtype=torch.int64)
    print(model(d))


#
# policy_network()

def trans_test():
    encoder_layer = nn.TransformerEncoderLayer(d_model=14, nhead=2, batch_first=True)
    src = torch.rand(3, 2, 7, 14)
    print(encoder_layer(src, is_causal=False))


# trans_test()

def gmt_test():
    gmt = pyg.nn.GraphMultisetTransformer(3,
                                          5,
                                          2,
                                          1)
    x = torch.tensor([[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]])
    index = torch.tensor([0, 2, 2])
    print(gmt(x, index))


# gmt_test()

def trans_dec_test():
    decoder_layer = nn.TransformerDecoderLayer(d_model=16, nhead=8)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    memory = torch.rand(2, 8, 16)
    tgt = torch.rand(2, 8, 16)
    print(transformer_decoder(tgt, memory))

trans_dec_test()