from models.alpha_zx_dist import AlphaZXDistribution
from models.utils import rand_azx_dist_params

# b = 1
# max_frz_nodes = 1
# max_flz_nodes = 0
# num_phases = 1
# num_new_edges = 1
# max_incident_edges = 1
# zero_prob = 0.0
# sample_size = 3
#
# mixture_dist_params, frz_node_dist_params, flz_node_dist_params, phase_dist_params, new_edges_dist_params, transfer_edges_dist_params = rand_azx_dist_params(
#     b, max_frz_nodes, max_flz_nodes, num_phases, num_new_edges, max_incident_edges, zero_prob)
# dist = AlphaZXDistribution(
#     mixture_dist_params, frz_node_dist_params, flz_node_dist_params, phase_dist_params, new_edges_dist_params,
#     transfer_edges_dist_params)

for b in range(1, 5):
    for max_frz_nodes in range(1, 10):
        for max_flz_nodes in range(10):
            for num_phases in range(1, 10):
                for num_new_edges in range(1, 10):
                    for max_incident_edges in range(1, 10):
                        mixture_dist_params, frz_node_dist_params, flz_node_dist_params, phase_dist_params, new_edges_dist_params, transfer_edges_dist_params = rand_azx_dist_params(
                            b, max_frz_nodes, max_flz_nodes, num_phases, num_new_edges, max_incident_edges, 0.05)
                        dist = AlphaZXDistribution(
                            mixture_dist_params, frz_node_dist_params, flz_node_dist_params, phase_dist_params,
                            new_edges_dist_params,
                            transfer_edges_dist_params)
                        samples = dist.sample(3)
                        probs = dist.log_prob(samples)
                        print('probs = ', probs.exp())
