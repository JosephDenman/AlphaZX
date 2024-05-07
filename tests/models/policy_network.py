import torch

from alphazx.diagram.diagram_generators import clifford_pyg_zx_match_diagram
from alphazx.diagram.match import NODE_METADATA, POSSIBLE_PHASES
from alphazx.distributions.alpha_zx_dist import AlphaZXDistribution
from alphazx.models.policy_network import PolicyNetwork
from alphazx.models.pre_process import with_embeddable_feats, with_laplacian_pe


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
    pooling_layer_norm = True
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
    for _ in range(10):
        d = clifford_pyg_zx_match_diagram(100, 100)
        d = with_embeddable_feats(d)
        d = with_laplacian_pe(d, 2)
        d.batch = torch.zeros(d.x.size(0), dtype=torch.int64)
        azx_dist = AlphaZXDistribution(model(d))
        print(azx_dist.sample(8))


policy_network()