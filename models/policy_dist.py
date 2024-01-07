import torch
from torch.distributions import Bernoulli, MixtureSameFamily, Categorical, Independent


class BernoulliMixture(MixtureSameFamily):
    def __init__(self,
                 mixture_logits: torch.Tensor = None,
                 component_logits: torch.Tensor = None,
                 mixture_probs: torch.Tensor = None,
                 component_probs: torch.Tensor = None):
        if (mixture_probs is None) == (mixture_logits is None):
            raise ValueError("Either `mixture_probs` or `mixture_logits` must be specified, but not both.")
        if (component_probs is None) == (component_logits is None):
            raise ValueError("Either `component_probs` or `component_logits` must be specified, but not both.")
        mixture_params = {'probs': mixture_probs} if mixture_probs is not None else {'logits': mixture_logits}
        component_params = {'probs': component_probs} if component_probs is not None else {'logits': component_logits}
        super().__init__(Categorical(**mixture_params), Independent(Bernoulli(**component_params), 1))


def recover_original_tensor_batch(padded_tensors: torch.Tensor) -> list[torch.Tensor]:
    # Recovering tensors assuming original dimensions of N x (N+1)
    # We need to find the first all-zero row to determine N, and assume N+1 for column length
    original_tensors = []
    for tensor in padded_tensors:
        # Find N, the number of non-zero rows which represents original height
        n = next((i for i, row in enumerate(tensor) if torch.all(row == 0)), tensor.shape[0])
        # Assuming the width is N+1, slice the tensor accordingly
        original_tensors.append(tensor[:n, :n + 1])
    return original_tensors


class FRZDist:

    def __init__(self,
                 frz_node_logits: torch.Tensor,
                 frz_phase_logits: torch.Tensor,
                 frz_n_edge_logits: torch.Tensor,
                 frz_o_edge_logits: torch.Tensor):
        """
        N = the number of existing nodes
        P = the number of buckets in the discretized phase space
        R = the max number of edges between an existing node and the new node connected to it
        S = the maximum degree across all existing nodes + 1

        N > 0 always (the game finishes when no nodes are left)
        P > 0 always (we never want just one phase to sample from)
        R > 0 always (fission must always produce a node connected to the original)
        S > 0 always (no isolated nodes)

        :param frz_node_logits: [N] of node selection probabilities
        :param frz_phase_logits: [N, P] of new phase probabilities
        :param frz_n_edge_logits: [N, R] of new edge probabilities
        :param frz_o_edge_logits: [N, S, S] of old edge probabilities
        """
        self.frz_node_logits = frz_node_logits
        self.frz_phase_logits = frz_phase_logits
        self.frz_n_edge_logits = frz_n_edge_logits
        self.frz_o_edge_logits = frz_o_edge_logits
        print('frz_node_logits = ', frz_node_logits)
        print('frz_phase_logits = ', frz_phase_logits)
        print('frz_n_edge_logits = ', frz_n_edge_logits)
        print('frz_o_edge_logits = ', frz_o_edge_logits)

    def sample(self, N: int = 1):
        size = torch.Size([N])
        node_samples = Categorical(logits=self.frz_node_logits).sample(size)  # B
        # print('self.frz_phase_logits[node_samples] = ', self.frz_phase_logits[node_samples])
        phase_samples = Categorical(logits=self.frz_phase_logits[node_samples]).sample(size)  # B x N
        n_edge_samples = Categorical(logits=self.frz_n_edge_logits[node_samples]).sample(size)  # B x N
        o_edge_logits = self.frz_o_edge_logits[node_samples]
        o_edge_mixture_logits = torch.squeeze(o_edge_logits[:, :, :1])
        # TODO: Special case N == 1 is not ideal
        o_edge_component_logits = torch.squeeze(o_edge_logits[:, :, 1:]) if N == 1 else o_edge_logits[:, :, 1:]
        # print('o_edge_mixture_logits = ', o_edge_mixture_logits)
        # print('o_edge_component_logits = ', o_edge_component_logits)
        o_edge_samples = BernoulliMixture(mixture_logits=o_edge_mixture_logits,
                                          component_logits=o_edge_component_logits).sample(size)
        return node_samples, phase_samples, n_edge_samples, o_edge_samples

    def log_prob(self,
                 node_samples: torch.Tensor,
                 phase_samples: torch.Tensor,
                 n_edge_samples: torch.Tensor,
                 o_edge_samples: torch.Tensor):
        print('node_samples = ', node_samples)
        node_samples_log_probs = Categorical(logits=self.frz_node_logits).log_prob(node_samples)
        return node_samples_log_probs
