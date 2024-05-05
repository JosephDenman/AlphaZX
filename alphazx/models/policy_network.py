from typing import Any

import torch
import torch.nn as nn
import torch_geometric as pyg
from torch import index_select

from alphazx import concatenate_with_neighbor_features, concatenate_neighbor_features
from alphazx.diagram.diagram_generators import clifford_pyg_zx_match_diagram
from alphazx.diagram.match import NODE_METADATA, POSSIBLE_PHASES, SIMPLE_NODE_METADATA, BoundaryMatch, \
    SIMPLE_EDGE_METADATA
from alphazx.distributions.alpha_zx_dist import AZXDistributionParams
from alphazx.models.attention import SigmoidCrossAttention
from alphazx.models.gps import GPS
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
                 node_embedding_channels,
                 gps_channels: int,
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
                 num_pooling_encoder_blocks: int,
                 num_pooling_heads: int,
                 pooling_layer_norm: bool,
                 pooling_dropout: float):
        super(PolicyNetwork, self).__init__()
        self.gps = GPS(num_node_types * num_possible_phases,
                       gps_channels,
                       node_embedding_channels,
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
        self.aggr = pyg.nn.GraphMultisetTransformer(node_embedding_channels, num_node_types, layer_norm=True)
        # self.node_logits_mlp = pyg.nn.MLP([node_embedding_channels, 1 + num_possible_phases + num_possible_new_edges])
        # self.neighbor_logits_mlp = pyg.nn.MLP([node_embedding_channels, 1])
        self.sigmoid_attn = SigmoidCrossAttention(node_embedding_channels)
        # self.pool = pyg.nn.GraphMultisetTransformer(1,
        #                                             1,
        #                                             num_pooling_encoder_blocks,
        #                                             num_pooling_heads,
        #                                             pooling_layer_norm,
        #                                             pooling_dropout)

    def _z_nodes(self, data: pyg.data.HeteroData) -> torch.Tensor:
        return self.z_mlp(data['z'].x)

    def forward(self, data: pyg.data.Data) -> AZXDistributionParams:
        """
        TODO: Figure out batching. All of the edge index based operations should stay correct, since the batched graphs
              are disconnected. We just have to collect the result from the different connected components using 'data.batch'.
        TODO: Ensure that removing connected components from the ZXDiagram does not affect 'Data' batching.
        :param data:
        :return: Parameters for the AlphaZXDistribution. Each value in the returned dictionary is a batch of distribution
                 parameters.
        """
        print('data = ', data)
        print('data.x = ', data.x)
        x = self.gps(data.x, data.pe, data.edge_index, data.edge_attr, data.batch)
        x = self.aggr(index_select(x, 0, data.edge_index[0]), data.edge_index[1])
        neighbor_x = concatenate_neighbor_features(x, data.edge_index)
        print('neighbor_x = ', neighbor_x)
        # x = concatenate_with_neighbor_features(x, data.edge_index)
        # print('data.x.concat = ', x)
        mask = (neighbor_x == torch.fill(torch.empty(x.size(-1)), -torch.inf).unsqueeze(0).unsqueeze(0)).all(dim=-1)
        transfer_edge_params = self.sigmoid_attn(x.unsqueeze(dim=1), neighbor_x, mask).squeeze(dim=1)
        print('attn_weights = ', attn_weights)
        # # x = self.encoder(x, src_key_padding_mask=mask, is_causal=False)
        # node_logits = self.node_logits_mlp(x[:, :1, :]).squeeze(dim=-2)
        # neighbor_logits = self.neighbor_logits_mlp(x[:, 1:, :]).squeeze(dim=-1).sigmoid()
        # neighbor_logits[mask[:, 1:]] = 0.
        # logits = torch.cat([node_logits, neighbor_logits], dim=-1)
        # print('logits = ', logits)
        # print('data.node_type = ', data.node_type)
        # mixture_dist_params = self.pool(logits, data.node_type)
        # print('mixture_dist_params = ', mixture_dist_params)


def policy_network():
    num_node_types = len(NODE_METADATA)
    num_possible_phases = len(POSSIBLE_PHASES)
    num_possible_new_edges = 15
    gps_channels = 1 + num_possible_phases + num_possible_new_edges + 2
    node_embedding_channels = 1 + num_possible_phases + num_possible_new_edges
    gps_edge_in_channels = 2
    gps_edge_out_channels = 2
    gps_pe_in_channels = 2
    gps_pe_out_channels = 2
    gps_num_layers = 5
    gps_bias = False
    gps_num_attn_heads = 1
    gps_attn_type = 'multihead'
    gps_attn_kwargs = {}
    gps_mlp_hidden_channels = 5
    num_pooling_encoder_blocks = 2
    num_pooling_heads = 1
    pooling_layer_norm = False
    pooling_dropout = 0.0
    model = PolicyNetwork(num_node_types,
                          num_possible_phases,
                          num_possible_new_edges,
                          node_embedding_channels,
                          gps_channels,
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
                          num_pooling_encoder_blocks,
                          num_pooling_heads,
                          pooling_layer_norm,
                          pooling_dropout)
    d = clifford_pyg_zx_match_diagram(10, 10)
    d = with_embeddable_feats(d)
    d = with_laplacian_pe(d, 2)
    d.batch = torch.zeros(d.x.size(0), dtype=torch.int64)
    print(model(d))


policy_network()


def compute_mask(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    # Compare x with v across the third dimension, and check if all elements along that dimension are True
    mask = (x == v.unsqueeze(0).unsqueeze(0)).all(dim=-1)
    return mask


def trans_enc_test():
    encoder_layer = nn.TransformerEncoderLayer(d_model=3, nhead=1, batch_first=True)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    node_logits_mlp = pyg.nn.MLP([3, 1 + 5 + 2])
    neighbor_logits_mlp = pyg.nn.MLP([3, 1])
    x = torch.tensor([[[1., 2., 3.], [4., 5., 6.], [0., 0., 0.]], [[1., 2., 3.], [4., 5., 6.], [7., 8., 9.]]])
    print('internal_shape = ', x.size(-1))
    mask = compute_mask(x, torch.tensor([0., 0., 0.]))
    print('mask = ', mask)
    x = transformer_encoder(x, src_key_padding_mask=mask, is_causal=False)
    print('x = ', x)
    node_embeddings = x[:, :1, :]
    print('node_embeddings = ', node_embeddings)
    neighbor_embeddings = x[:, 1:, :]
    print('neighbor_embeddings = ', neighbor_embeddings)
    node_logits = node_logits_mlp(node_embeddings).squeeze(dim=-2)
    print('node_logits = ', node_logits)
    # print('squeezed_node_logits = ', node_logits.squeeze(dim=-2))
    neighbor_logits = neighbor_logits_mlp(neighbor_embeddings).squeeze(dim=-1)
    print('neighbor_logits = ', neighbor_logits)
    neighbor_logits[mask[:, 1:]] = 0.
    print('masked_neighbor_logits = ', neighbor_logits)
    final_node_logits = torch.cat([node_logits, neighbor_logits], dim=-1)
    # x = torch.softmax(x, dim=1)
    print(final_node_logits)


# trans_enc_test()


def sab_test():
    data = clifford_pyg_zx_match_diagram(10, 10)
    print('data.x = ', data.x)
    print('data.x.size = ', data.x.size())
    print('data.edge_index = ', data.edge_index)
    sab = pyg.nn.DeepSetsAggregation()
    neighbor_x = index_select(data.x, 0, data.edge_index[0])
    result = sab(neighbor_x, data.edge_index[1])
    print('result = ', result)
    print('result.size = ', result.size())


# sab_test()


def mlp_aggr_test():
    data = clifford_pyg_zx_match_diagram(10, 10)
    print('data.x = ', data.x)
    print('data.x.size = ', data.x.size())
    print('data.edge_index = ', data.edge_index)
    mlp_aggr = pyg.nn.MLPAggregation(in_channels=2, out_channels=1, num_layers=4, hidden_channels=5,
                                     max_num_elements=500)
    neighbor_x = index_select(data.x, 0, data.edge_index[0])
    result = mlp_aggr(neighbor_x, data.edge_index[1])
    print('result = ', result)
    print('result.size = ', result.size())


def logistic(x: torch.Tensor) -> torch.Tensor:
    return (1. / (1. + torch.exp(-10. * x)))


def mha_test():
    mha = nn.MultiheadAttention(embed_dim=3, kdim=3, vdim=3, num_heads=1, batch_first=True)
    queries = torch.tensor([[[1., 2., 3.]], [[7., 8., 9.]]])
    keys = torch.tensor([[[4., 5., 6.], [0., 0., 0.]], [[10., 11., 12.], [13., 14., 15.]]])
    attn_mask = torch.tensor([[False, True], [False, False]])
    # layer_norm = nn.LayerNorm(3)
    # queries = layer_norm(queries)
    # keys = layer_norm(keys)
    # This works! Use it for neighborhood aggregation!
    _, attn_weights = mha(queries, keys, keys, key_padding_mask=attn_mask, is_causal=False, need_weights=True)
    attn_weights = attn_weights.squeeze(dim=1)
    print(attn_weights)
    attn_weights = torch.sigmoid(attn_weights)
    attn_weights[attn_mask] = 0.
    print(attn_weights)


# mha_test()

def custom_attn_test():
    attn = SigmoidCrossAttention(3)
    queries = torch.tensor([[[1., 2., 3.]], [[7., 8., 9.]]])
    keys = torch.tensor([[[4., 5., 6.], [0., 0., 0.]], [[10., 11., 12.], [13., 14., 15.]]])
    mask = torch.tensor([[False, True], [False, False]])
    # layer_norm = nn.LayerNorm(3)
    # queries = layer_norm(queries)
    # keys = layer_norm(keys)
    # This works! Use it for neighborhood aggregation!
    attn_weights = attn(queries, keys, mask)
    print(attn_weights)
    # layer_norm = nn.LayerNorm(2)
    # attn_weights = layer_norm(attn_weights)
    # print(attn_weights)
    # print(torch.sigmoid(attn_weights))


# custom_attn_test()

def gmt_test():
    data = clifford_pyg_zx_match_diagram(10, 10)
    print('data.x = ', data.x)
    print('data.x.size = ', data.x.size())
    print('data.edge_index = ', data.edge_index)
    gmt = pyg.nn.GraphMultisetTransformer(2,
                                          5,
                                          2,
                                          1)
    neighbor_x = index_select(data.x, 0, data.edge_index[0])
    result = gmt(neighbor_x, data.edge_index[1])
    print('result = ', result)
    print('result.size = ', result.size())


# gmt_test()


def trans_dec_test():
    decoder_layer = nn.TransformerDecoderLayer(d_model=16, nhead=8)
    transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
    memory = torch.rand(2, 8, 16)
    tgt = torch.rand(2, 8, 16)
    print(transformer_decoder(tgt, memory))


# trans_dec_test()

# def concatenate_neighbor_features(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
#     neighbor_x = index_select(x, 0, edge_index[0])
#     x_ = concatenate_by_group(neighbor_x, edge_index[1])
#     return x_

def sta_test():
    data = clifford_pyg_zx_match_diagram(10, 10)
    print('data.x = ', data.x)
    print('data.x.size = ', data.x.size())
    print('data.edge_index = ', data.edge_index)
    sta = pyg.nn.SetTransformerAggregation(2)
    neighbor_x = index_select(data.x, 0, data.edge_index[0])
    result = sta(neighbor_x, data.edge_index[1])
    print('result = ', result)
    print('result.size = ', result.size())

# sta_test()
