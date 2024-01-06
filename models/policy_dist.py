import torch
import torch.nn.functional as F
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


# batch_count = 6
# component_count = 5
# variable_count = 5
# # Note as logit bound increases, all-one bit strings are more likely.
# logit_bound = 10
# mix_probs = torch.softmax(torch.ones(batch_count, component_count), dim=-1)
# comp_probs = torch.softmax(torch.ones(batch_count, component_count, variable_count), dim=-1)
# bmm = BernoulliMixture(mixture_probs=mix_probs, component_probs=comp_probs)
# print('sample = ', bmm.sample())

def recover_original_tensor(padded_tensor: torch.Tensor) -> torch.Tensor:
    # Find the index of the first all-zero row to determine the size of the original tensor
    N = next((i for i, row in enumerate(padded_tensor) if torch.all(row == 0)), padded_tensor.shape[0])
    # Slice to recover the original tensor
    original_tensor = padded_tensor[:N, :N]
    return original_tensor


def recover_original_tensor_batch(padded_tensors: torch.Tensor) -> list[torch.Tensor]:
    # Assuming all tensors in the batch are padded to the same maximum dimensions
    # Determine the original dimensions of each tensor before padding
    # This would be the first all-zero row and column for each tensor

    # Step 1: Find the size of the original tensor for each in the batch
    n_batch = [next((i for i, row in enumerate(tensor) if torch.all(row == 0)), tensor.shape[0]) for tensor in
               padded_tensors]

    # Step 2: Recover each tensor based on its original size
    tmp = []
    for tensor, n in zip(padded_tensors, n_batch):
        print('n = ', n)
        print('t = ', tensor)
        print(f't[:{n},:{n}+1 = ', tensor[:n, :n+1])
    original_tensors = [tensor[:n, :n+1] for tensor, n in zip(padded_tensors, n_batch)]
    return original_tensors


class FRZDist:

    def __init__(self,
                 frz_node_logits: torch.Tensor,
                 frz_phase_logits: torch.Tensor,
                 frz_nedge_logits: torch.Tensor,
                 frz_oedge_logits: torch.Tensor):
        """
        N = the number of existing nodes
        P = the number of buckets in the discretized phase space
        R = the max number of edges between an existing node and the new node connected to it
        S = the maximum degree across all existing nodes + 1

        :param frz_node_logits: [N] of node selection probabilities
        :param frz_phase_logits: [N, P] of new phase probabilities
        :param frz_nedge_logits: [N, R] of new edge probabilities
        :param frz_oedge_logits: [N, S, S] of old edge probabilities
        """
        self.frz_node_logits = frz_node_logits
        self.frz_phase_logits = frz_phase_logits
        self.frz_nedge_logits = frz_nedge_logits
        self.frz_oedge_logits = frz_oedge_logits

        # self.frz_node_dist = Categorical(logits=frz_node_logits)
        # self.frz_phase_dist = Categorical(logits=frz_phase_logits)
        # self.frz_nedge_dist = Categorical(logits=frz_nedge_logits)
        # self.frz_oedge_dist = BernoulliMixture()

    def sample(self, n: int = 1):
        size = torch.Size([n])
        node_samples = Categorical(logits=self.frz_node_logits).sample(size)
        # print('node_samples = ', node_samples)
        phase_samples = Categorical(logits=torch.squeeze(self.frz_phase_logits[node_samples])).sample(size)
        # print('phase_samples = ', phase_samples)
        n_edge_samples = Categorical(logits=torch.squeeze(self.frz_nedge_logits[node_samples])).sample(size)
        # print('n_edge_samples = ', n_edge_samples)
        o_edge_logits = recover_original_tensor(torch.squeeze(self.frz_oedge_logits[node_samples]))
        o_edge_mixture_logits, o_edge_component_logits = torch.squeeze(o_edge_logits[:, :1]), o_edge_logits[:, 1:]
        print('o_edge_logits = ', o_edge_logits)
        # print('o_edge_mixture_logits = ', o_edge_mixture_logits)
        # print('o_edge_component_logits = ', o_edge_component_logits)
        o_edge_samples = BernoulliMixture(mixture_logits=o_edge_mixture_logits, component_logits=o_edge_component_logits).sample(size)
        # print('o_edge_samples = ', o_edge_samples)
        return node_samples, phase_samples, n_edge_samples, o_edge_samples

    def log_prob(self, sampled_frz_actions: torch.Tensor):
        # sampled_frz_action = [
        #   [one_hot_phase_categorical],
        #   [one_hot_new_edge_categorical],
        #   [one_hot_node_categorical],
        #   [bernoulli mixture params]
        # ]
        pass


nodes = 5
phase_buckets = 2
n_edge_buckets = 4
o_edge_buckets = 3

node_logits = torch.rand(torch.Size([nodes]))
# print('node_logits = ', node_logits)

phase_logits = torch.rand(torch.Size([nodes, phase_buckets]))
# print('phase_logits = ', phase_logits)

n_edge_logits = torch.rand(torch.Size([nodes, n_edge_buckets]))
# print('n_edge_logits = ', n_edge_logits)

o_edge_logits_list = []
for _ in range(nodes):
    o_edge_count = torch.randint(0, o_edge_buckets + 1, torch.Size([1])).item()
    print('o_edge_count = ', o_edge_count)
    o_edge_padding = o_edge_buckets - o_edge_count
    print('o_edge_padding = ', o_edge_padding)
    o_edge_logits_list.append(F.pad(torch.rand(torch.Size([o_edge_count, o_edge_count + 1])),
                                    (0, o_edge_padding, 0, o_edge_padding + 1)))

o_edge_logits = torch.stack(o_edge_logits_list)
print('o_edge_logits = ', o_edge_logits)

frz_dist = FRZDist(
    frz_node_logits=node_logits,
    frz_phase_logits=phase_logits,
    frz_nedge_logits=n_edge_logits,
    frz_oedge_logits=o_edge_logits)

print('sample = ', frz_dist.sample(2))
