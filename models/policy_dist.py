import torch
import torch.nn.functional as F
from torch.distributions import Bernoulli, MixtureSameFamily, Categorical, Independent

from models.utils import rand_node_probs, rand_phase_probs, rand_new_edge_probs, rand_o_edge_probs, \
    replace_zero_rows_with_uniform


class BernoulliMixture(MixtureSameFamily):
    """
    TODO: When all an all zeros [n, n+1] probability tensor is passed here, it causes an error due to reinterpreting batch
          dimensions. This situation is the situation in which there is zero probability that *any* old edge is selected
          to move to the new node.
    TODO: All zero mixture logits are a problem.
    """
    def __init__(self,
                 mixture_probs: torch.Tensor = None,
                 component_probs: torch.Tensor = None):
        # When all mixture weights are zero,
        mixture_probs = replace_zero_rows_with_uniform(mixture_probs)
        super().__init__(Categorical(probs=mixture_probs), Independent(Bernoulli(probs=component_probs), 1))


class FRZDist:

    def __init__(self,
                 frz_node_probs: torch.Tensor,
                 frz_phase_probs: torch.Tensor,
                 frz_n_edge_probs: torch.Tensor,
                 frz_o_edge_probs: torch.Tensor):
        """
        N = the number of existing nodes
        P = the number of buckets in the discretized phase space
        R = the max number of edges between an existing node and the new node connected to it
        S = the maximum degree across all existing nodes + 1

        N > 0 always (the game finishes when no nodes are left)
        P > 0 always (we never want just one phase to sample from)
        R > 0 always (fission must always produce a node connected to the original)
        S > 0 always (no isolated nodes)

        :param frz_node_probs: [N] of node selection probabilities
        :param frz_phase_probs: [N, P] of new phase probabilities
        :param frz_n_edge_probs: [N, R] of new edge probabilities
        :param frz_o_edge_probs: [N, S, S] of old edge probabilities
        """
        self.frz_node_probs = frz_node_probs
        self.frz_phase_probs = frz_phase_probs
        self.frz_n_edge_probs = frz_n_edge_probs
        self.frz_o_edge_probs = frz_o_edge_probs
        # print('frz_node_probs = ', frz_node_probs)
        # print('frz_phase_probs = ', frz_phase_probs)
        # print('frz_n_edge_probs = ', frz_n_edge_probs)
        # print('frz_o_edge_probs = ', frz_o_edge_probs)

    def sample(self, N: int = 1):
        size = torch.Size([N])
        node_samples = Categorical(probs=self.frz_node_probs).sample(size)  # B
        phase_samples = Categorical(probs=self.frz_phase_probs[node_samples]).sample(size)  # B x N
        n_edge_samples = Categorical(probs=self.frz_n_edge_probs[node_samples]).sample(size)  # B x N
        o_edge_probs = self.frz_o_edge_probs[node_samples]
        o_edge_mixture_probs = o_edge_probs[:, :, :1].squeeze(-1)
        # TODO: Special case N == 1 is not ideal
        o_edge_component_probs = o_edge_probs[:, :, 1:]
        print('node_samples = ', node_samples)
        print('phase_samples = ', phase_samples)
        print('n_edge_samples = ', n_edge_samples)
        print('\n')
        print('o_edge_mixture_probs = ', o_edge_mixture_probs)
        print('o_edge_component_probs = ', o_edge_component_probs)
        o_edge_samples = BernoulliMixture(mixture_probs=o_edge_mixture_probs, component_probs=o_edge_component_probs).sample(size)
        print('o_edge_probs = ', o_edge_probs)
        return node_samples, phase_samples, n_edge_samples, o_edge_samples

    def log_probability(self,
                        node_samples: torch.Tensor,
                        phase_samples: torch.Tensor,
                        n_edge_samples: torch.Tensor,
                        o_edge_samples: torch.Tensor) -> torch.Tensor:
        print('node_samples = ', node_samples)
        print('phase_samples = ', phase_samples)
        print('n_edge_samples = ', n_edge_samples)
        print('o_edge_samples = ', o_edge_samples)
        node_samples_log_probs = Categorical(probs=self.frz_node_probs).log_prob(node_samples)
        print('node_samples_probs = ', node_samples_log_probs.exp())
        phase_log_probs = Categorical(probs=self.frz_phase_probs[node_samples]).log_prob(phase_samples)  # B x N
        print('phase_samples_probs = ', phase_log_probs.exp())
        n_edge_log_probs = Categorical(probs=self.frz_n_edge_probs[node_samples]).log_prob(n_edge_samples)  # B x N
        print('n_edge_samples_probs = ', n_edge_log_probs.exp())
        o_edge_mixture_probs = torch.squeeze(self.frz_o_edge_probs[:, :, :1])
        o_edge_component_probs = self.frz_o_edge_probs[:, :, 1:]
        # TODO: BEGIN AGAIN HERE - NEED TO STRIP ALL ZEROES FROM MIXTURE AND COMPONENT PROBS BEFORE BUILDING THIS DIST
        o_edge_log_probs = BernoulliMixture(mixture_probs=o_edge_mixture_probs,
                                            component_probs=o_edge_component_probs).log_prob(o_edge_samples)
        print('o_edge_probs = ', o_edge_log_probs.exp())
        return node_samples_log_probs + phase_log_probs + n_edge_log_probs + o_edge_log_probs

    def probability(self,
                    node_samples: torch.Tensor,
                    phase_samples: torch.Tensor,
                    n_edge_samples: torch.Tensor,
                    o_edge_samples: torch.Tensor) -> torch.Tensor:
        return self.log_probability(node_samples, phase_samples, n_edge_samples, o_edge_samples).exp()
