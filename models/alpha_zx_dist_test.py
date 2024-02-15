from models.alpha_zx_dist import AlphaZXDistribution
from models.utils import rand_azx_dist_params

b = 2
max_frz_nodes = 5
max_flz_nodes = 5
num_phases = 4
num_new_edges = 6
max_incident_edges = 4
zero_prob = 0.05
sample_size = 3

mixture_dist_params, frz_node_dist_params, flz_node_dist_params, phase_dist_params, new_edges_dist_params, transfer_edges_dist_params = rand_azx_dist_params(
    b, max_frz_nodes, max_flz_nodes, num_phases, num_new_edges, max_incident_edges, zero_prob)
dist = AlphaZXDistribution(
    mixture_dist_params, frz_node_dist_params, flz_node_dist_params, phase_dist_params, new_edges_dist_params,
    transfer_edges_dist_params)

samples = dist.sample(sample_size)
print('samples =', samples)
dist.log_prob(samples)
