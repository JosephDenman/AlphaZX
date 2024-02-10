import torch
from torch.distributions.categorical import Categorical

from models.bernoulli_mixture import MultivariateBernoulliMixture


def gather_probs(probs_batch: torch.Tensor, sampled_actions_batch: torch.Tensor, column: int) -> torch.Tensor:
    """
    Uses the column'th entry of each sampled action to get the probability of the entry occurring.
    Works with mini-batching.

    :param probs_batch: B x X tensor of probabilities, where X = T or X = N.
    :param sampled_actions_batch: B x K x L tensor of actions.
    :param column: The row entry in `sampled_actions_batch` to use to index into probabilities.
    :return: B x X tensor representing the probability of the given row entry.
    """
    return torch.gather(probs_batch, 1, sampled_actions_batch[:, :, column].long())


def gather_mixture_probs(mixture_probs_batch: torch.Tensor, sampled_actions_batch: torch.Tensor) -> torch.Tensor:
    return gather_probs(mixture_probs_batch, sampled_actions_batch, 0)


def gather_node_probs(node_probs_batch: torch.Tensor, sampled_actions_batch: torch.Tensor) -> torch.Tensor:
    return gather_probs(node_probs_batch, sampled_actions_batch, 1)


def gather_phase_probs(phase_probs_batch: torch.Tensor, sampled_actions_batch: torch.Tensor) -> torch.Tensor:
    """
    Extracts the probability of a particular phase for each action based on the second and third
    row entries of the sampled actions, using a vectorized approach with torch.gather.

    :param phase_probs_batch: A tensor of shape (batch_size, num_nodes, num_phase_buckets) containing the probabilities.
    :param sampled_actions_batch: A tensor of shape (batch_size, num_actions, action_length) containing the sampled actions.
                                  The second row entry corresponds to the node index, and the third row entry corresponds to the phase index.
    :returns: A tensor of probabilities extracted for each action.
    """
    # Extract node and phase indices
    node_indices = sampled_actions_batch[:, :, 1].long()
    phase_indices = sampled_actions_batch[:, :, 2].long()
    # First gather along the num_nodes dimension to get [batch_size, num_actions, num_phase_buckets]
    gathered_nodes = torch.gather(phase_probs_batch, 1, node_indices.unsqueeze(-1).expand(-1, -1, phase_probs_batch.size(2)))
    # Then gather along the last dimension to select the specific phase for each action
    phase_probs_selected = torch.gather(gathered_nodes, 2, phase_indices.unsqueeze(-1)).squeeze(-1)
    return phase_probs_selected


class AlphaZXDistribution:
    """
    Currently, limited to two rewrite types: frz and flz, represented as 0 and 1 in the first entry of an action tensor.
    In the following:

        B = batch size
        K = number of samples
        L = length of the longest action (either in the batch or sample set)
        T = number of possible rewrites (frz, flz, frx, flx, etc.)
        N = max number of nodes (either in batch or sample set)
        P = number of phase buckets
        E_new = number of new edge buckets
        E_trans = max degree of any node in the batch

    For simplicity, T is always fixed to the maximum (two in this case).
    """
    def __init__(self,
                 mixture_dist_parameters: torch.Tensor,
                 flz_node_dist_parameters: torch.Tensor,
                 frz_node_dist_parameters: torch.Tensor,
                 phase_dist_parameters: torch.Tensor,
                 new_edges_dist_parameters: torch.Tensor,
                 transfer_edges_dist_parameters: torch.Tensor):
        """
        :param mixture_dist_parameters: B x T tensor of mixture probabilities. mixture_probs_batch[b] is the mixture probabilities
                                        at some step in a trajectory. It is assumed to be a softmax output of a DNN. When B = 1,
                                        we are in the MCTS portion of the algorithm.
        :param flz_node_dist_parameters: B x N tensor of node probabilities. For a node n in batch b, that does not represent an f-left match,
                                         flz_node_dist_parameters[b, n] = 0.
        :param frz_node_dist_parameters: B x N tensor of node probabilities. For a node n in batch b, that does not represent an f-right match,
                                         frz_node_dist_parameters[b, n] = 0.
        :param phase_dist_parameters: B x N x P tensor of phase probabilities. For a node n in batch b, that does not
                                      represent an f-right match, phase_prob_parameters[b, n] = torch.zeroes((P, ))
        :param new_edges_dist_parameters: B x N x E_new tensor of new edge probabilities. For a node n in batch b that does not
                                          represent an f-right match, new_edges_prob_parameters_batch[b, n] is all zeros.
        :param transfer_edges_dist_parameters: B x N x E_trans x (E_trans + 1) tensor of probabilities for each node. For a node n in batch b
                                               that does not represent an f-right match, new_edges_prob_parameters_batch[b, n] has a single 1 entry
                                               in the top left corner of the innermost 2D tensor. All other entries in the innermost 2D tensor are 0.
                                               Samples drawn from this distribution are all zeros (no edges are selected to be transferred).
        """
        self.mixture_dist = Categorical(mixture_dist_parameters)
        self.flz_node_dist = Categorical(flz_node_dist_parameters)
        self.frz_node_dist = Categorical(frz_node_dist_parameters)
        self.phase_dist = Categorical(phase_dist_parameters)
        self.new_edges_dist = Categorical(new_edges_dist_parameters)
        self.transfer_edges_dist = MultivariateBernoulliMixture(transfer_edges_dist_parameters)

    def log_prob(self, sampled_actions: torch.Tensor) -> torch.Tensor:
        """
        :param sampled_actions: B x K x L tensor of actions. Each action has the form
                                [type, node, phase, new edges, old edges ...], where all entries after node are 0
                                for non-f-right actions. All actions are padded to the same length with zeroes.
                                sampled_actions_batch[b] is the set of actions sampled at some step in a trajectory.
        :return: The log probability of each sampled action.
        """
        pass

    def sample(self, k: int) -> torch.Tensor:
        """
        :param k: The number of samples to produce.
        :return: K x L tensor of actions.
        """
        rewrite_type_samples = self.mixture_dist.sample(torch.Size([k]))
