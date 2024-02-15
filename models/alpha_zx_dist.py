import torch
from torch.distributions.categorical import Categorical

from models.bernoulli_mixture import MultivariateBernoulliMixture


def action_types_log_prob(mixture_dist_params: torch.Tensor, action_types: torch.Tensor) -> torch.Tensor:
    return torch.gather(mixture_dist_params.log(), 1, action_types)


def nodes_log_prob(frz_node_dist_params: torch.Tensor, flz_node_dist_params: torch.Tensor, action_types: torch.Tensor,
                   nodes: torch.Tensor) -> torch.Tensor:
    selected_node_dist_params = select_node_dist_params(frz_node_dist_params, flz_node_dist_params,
                                                        action_types)
    batch_size, num_nodes = nodes.shape
    # Generate a tensor of batch indices to pair with each node index
    batch_indices = torch.arange(batch_size).view(-1, 1).expand_as(nodes)
    # Use advanced indexing to select the corresponding distribution parameters for each node
    nodes_log_prob = selected_node_dist_params.log()[batch_indices, torch.arange(num_nodes), nodes]
    return nodes_log_prob


def select_node_dist_params(frz_node_dist_params: torch.Tensor, flz_node_dist_params: torch.Tensor,
                            action_types: torch.Tensor) -> torch.Tensor:
    # Reshape action_types for broadcasting over the distributions
    action_types_expanded = action_types.unsqueeze(-1).expand(-1, -1, frz_node_dist_params.size(-1))
    # Use torch.where to select between the two parameter sets based on action types
    selected_node_dist_params = torch.where(action_types_expanded == 0,
                                            frz_node_dist_params.unsqueeze(1).repeat(1, action_types.size(1), 1),
                                            flz_node_dist_params.unsqueeze(1).repeat(1, action_types.size(1), 1))
    return selected_node_dist_params


def sample_from_selected_dist_params(selected_dist_params: torch.Tensor) -> torch.Tensor:
    # Flatten the tensor from [batch_size, num_distributions, num_classes] to [batch_size * num_distributions, num_classes]
    # This is necessary because Categorical treats the first dimension as the batch dimension
    flattened_distributions = selected_dist_params.view(-1, selected_dist_params.size(-1))
    # Create the Categorical distribution
    dist = Categorical(probs=flattened_distributions)
    # To sample or compute probabilities, use the distribution as usual
    # For example, to sample one set of events for each distribution:
    samples = dist.sample()
    # Reshape the samples back to the original [batch_size, num_distributions] format
    reshaped_samples = samples.view(selected_dist_params.size(0), selected_dist_params.size(1))
    return reshaped_samples


def select_feature_dist_params(nodes: torch.Tensor, feature_dist_params: torch.Tensor) -> torch.Tensor:
    """
    Selects rows from feature_dist_params based on indices in nodes while respecting batching.

    :param nodes: A tensor of indices indicating which rows to select from phase_dist_params.
    :param feature_dist_params: A tensor containing parameters for different phases or new edge features, with batching.
    :returns: A tensor with selected distributions based on nodes.
    """
    # Obtain batch indices for each element in nodes to use with advanced indexing
    batch_indices = torch.arange(nodes.size(0)).view(-1, 1).expand_as(nodes)
    # Use advanced indexing to select the rows from phase_dist_params
    selected_distributions = feature_dist_params[batch_indices, nodes]
    return selected_distributions


class AlphaZXDistribution:
    """
    Currently, limited to two rewrite types: frz and flz, represented as 0 and 1 in the first entry of an action tensor.
    In the following:

        B = batch size
        K = number of samples
        L = length of the longest action (either in the batch or sample set)
        T = number of possible rewrites (frz, flz, frx, flx, etc.)
        N = max number of nodes across node types across batches, or across samples
        P = number of phase buckets
        E_new = number of new edge buckets
        E_trans = max degree of any node in the batch

    For simplicity, T is always fixed to the maximum (two in this case).
    """

    def __init__(self,
                 mixture_dist_params: torch.Tensor,
                 frz_node_dist_params: torch.Tensor,
                 flz_node_dist_params: torch.Tensor,
                 phase_dist_params: torch.Tensor,
                 new_edges_dist_params: torch.Tensor,
                 transfer_edges_dist_params: torch.Tensor):
        """
        :param mixture_dist_params: B x T tensor of mixture probabilities. mixture_dist_params[b] is the mixture probabilities
                                    at some step in a trajectory. It is assumed to be a softmax output of a DNN. When B = 1,
                                    we are in the MCTS portion of the algorithm.
        :param frz_node_dist_params: B x N tensor of frz-node selection probabilities. For a node n in batch b that does
                                     not represent a frz-node, frz_node_dist_params[b, n] = 0.
        :param frz_node_dist_params: B x N tensor of flz-node selection probabilities. For a node n in batch b that does
                                     not represent a flz-node, flz_node_dist_params[b, n] = 0.
        :param phase_dist_params: B x N x P tensor of phase probabilities. For a node n in batch b that does not
                                  represent an f-right match, phase_dist_params[b, n] = torch.zeroes((P, ))
        :param new_edges_dist_params: B x N x E_new tensor of new edge probabilities. For a node n in batch b that does not
                                      represent an f-right match, new_edges_dist_params[b, n] is all zeros.
        :param transfer_edges_dist_params: B x N x E_trans x (E_trans + 1) tensor of probabilities for each node. For a node n in batch b
                                           that does not represent an f-right match, transfer_edges_dist_params[b, n] has a single 1 entry
                                           in the top left corner of the innermost 2D tensor. All other entries in the innermost 2D tensor are 0.
                                           Samples drawn from this distribution are all zeros (no edges are selected to be transferred).
        """
        self.mixture_dist_params = mixture_dist_params
        self.frz_node_dist_params = frz_node_dist_params
        self.flz_node_dist_params = flz_node_dist_params
        self.phase_dist_params = phase_dist_params
        self.new_edges_dist_params = new_edges_dist_params
        self.transfer_edges_dist_params = transfer_edges_dist_params
        print('mixture_dist_params =', mixture_dist_params)
        print('frz_node_dist_params =', frz_node_dist_params)
        print('flz_node_dist_params =', flz_node_dist_params)
        # print('phase_dist_params =', phase_dist_params)
        # print('new_edges_dist_params =', new_edges_dist_params)
        # print('transfer_edges_dist_params =', transfer_edges_dist_params)
        print('')

    def log_prob(self, sampled_actions: torch.Tensor) -> torch.Tensor:
        """
        :param sampled_actions: B x K x L tensor of actions. Each action has the form
                                [type, node, phase, new edges, old edges ...], where all entries after node are 0
                                for non-f-right actions. All actions are padded to the same length with zeroes.
                                sampled_actions_batch[b] is the set of actions sampled at some step in a trajectory.
        :return: The log probability of each sampled action.
        """
        action_types = sampled_actions[:, :, 0]
        print('action_types =', action_types)
        action_type_log_probs = action_types_log_prob(self.mixture_dist_params, action_types)
        print('action_type_log_probs =', action_type_log_probs.exp())
        nodes = sampled_actions[:, :, 1]
        print('nodes = ', nodes)
        node_log_probs = nodes_log_prob(self.frz_node_dist_params, self.flz_node_dist_params, action_types, nodes)
        print('node_log_probs =', node_log_probs.exp())
        phases = sampled_actions[:, :, 2]
        new_edges = sampled_actions[:, :, 3]
        transfer_edges = sampled_actions[:, :, 4:]

    def sample(self, k: int) -> torch.Tensor:
        """
        :param k: The number of samples to produce.
        :return: K x L tensor of actions.
        """
        action_types = Categorical(probs=self.mixture_dist_params).sample(torch.Size([k])).T
        selected_node_dist_params = select_node_dist_params(self.frz_node_dist_params, self.flz_node_dist_params,
                                                            action_types)
        nodes = sample_from_selected_dist_params(selected_node_dist_params)
        selected_phase_dist_params = select_feature_dist_params(nodes, self.phase_dist_params)
        phases = sample_from_selected_dist_params(selected_phase_dist_params)
        selected_new_edges_dist_params = select_feature_dist_params(nodes, self.new_edges_dist_params)
        new_edges = sample_from_selected_dist_params(selected_new_edges_dist_params)
        selected_transfer_edges_dist_params = select_feature_dist_params(nodes, self.transfer_edges_dist_params)
        transfer_edges = MultivariateBernoulliMixture(selected_transfer_edges_dist_params).sample()
        return torch.cat((torch.stack((action_types, nodes, phases, new_edges), dim=-1), transfer_edges), dim=-1).long()

# Define the mixture distribution parameters
# mixture_dist_params = torch.tensor([[0.3157, 0.6843],
#                                     [0.3269, 0.6731]])
#
# # Define the action types for which you want to compute log probabilities
# action_types = torch.tensor([[1., 1., 0.],
#                              [1., 1., 1.]])
#
# log_mixture_dist_params = torch.log(mixture_dist_params)
#
# # Prepare to gather the log probabilities based on action_types
# # Convert action_types to long and add extra dimensions for gathering
# action_types_long = action_types.long()
#
# gathered_log_probs = torch.gather(log_mixture_dist_params, 1, action_types_long).exp()
#
# print("Gathered log probabilities:")
# print(gathered_log_probs)

# nodes =  tensor([[3, 2, 0],
#         [3, 2, 3]])
# selected_node_dist_params = tensor([[[0.4390, 0.2160, 0.1784, 0.1371, 0.0294, 0.0000],
#          [0.4390, 0.2160, 0.1784, 0.1371, 0.0294, 0.0000],
#          [0.4390, 0.2160, 0.1784, 0.1371, 0.0294, 0.0000]],
#
#         [[0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000],
#          [0.3706, 0.1567, 0.2055, 0.0000, 0.0354, 0.2318],
#          [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000]]])
#
# torch.tensor([[0.1371, 0.1784, 0.4390], [1.0000, 0.2055, 1.0000]])
