import torch

from models.utils import rand_mixture_probs, rand_node_probs, insert_random_zeros

b = 1
num_frz_nodes = 5
num_flz_nodes = 5
num_action_types = 2
num_phase_buckets = 3

mixture_dist_parameters = rand_mixture_probs(b, num_action_types)
flz_node_dist_parameters = insert_random_zeros(rand_node_probs(b, num_flz_nodes).squeeze(-1), num_frz_nodes)

frz_node_dist_parameters = torch.zeros_like(flz_node_dist_parameters)
frz_node_dist_parameters[torch.nonzero(flz_node_dist_parameters == 0).squeeze()] = rand_node_probs(1, num_frz_nodes).squeeze(-1)
print('frz_node_dist_parameters = ', frz_node_dist_parameters)
print('flz_node_dist_parameters = ', flz_node_dist_parameters)
# flz_node_dist_parameters
# frz_node_dist_parameters
# phase_dist_parameters
# new_edges_dist_parameters
# transfer_edges_dist_parameter