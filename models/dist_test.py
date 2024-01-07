import unittest
import torch
from torch.distributions import Categorical

from models.policy_dist import FRZDist, BernoulliMixture
import torch.nn.functional as F


def rand_float(size: torch.Size, low: int = 0, high: int = 100) -> torch.Tensor:
    return torch.FloatTensor(size).uniform_(low, high)


def rand_node_logits(nodes: int) -> torch.Tensor:
    return rand_float(torch.Size([nodes]))


def rand_phase_logits(nodes: int, phase_buckets: int) -> torch.Tensor:
    return rand_float(torch.Size([nodes, phase_buckets]))


def rand_n_edge_logits(nodes: int, n_edge_buckets: int) -> torch.Tensor:
    return rand_float(torch.Size([nodes, n_edge_buckets]))


def rand_o_edge_logits(nodes: int, o_edge_buckets: int) -> torch.Tensor:
    o_edge_logits = []
    for _ in range(nodes):
        o_edge_count = torch.randint(0, o_edge_buckets + 1, torch.Size([1])).item()
        o_edge_padding = o_edge_buckets - o_edge_count
        o_edge_logits.append(F.pad(rand_float(torch.Size([o_edge_count, o_edge_count + 1])),
                                   (0, o_edge_padding, 0, o_edge_padding + 1)))
    return torch.stack(o_edge_logits)


class DistTest(unittest.TestCase):

    def test_dist(self):
        for n in range(1, 6):
            for pb in range(1, 6):
                for neb in range(1, 6):
                    for oeb in range(1, 6):
                        d = FRZDist(
                            frz_node_logits=rand_node_logits(n),
                            frz_phase_logits=rand_phase_logits(n, pb),
                            frz_n_edge_logits=rand_n_edge_logits(n, neb),
                            frz_o_edge_logits=rand_o_edge_logits(n, oeb))
                        for k in range(2, 6):
                            samples = d.sample(k)
                            print('probs = ', d.log_prob(*samples))

    def test_cat(self):
        frz_node_logits = torch.tensor([19.4782])
        node_samples = torch.tensor([0, 0])
        probs = torch.tensor([0., 0.])
        dist = Categorical(logits=frz_node_logits)
        print('tmp_prob = ', dist.log_prob(0))
        print('sample = ', dist.sample(torch.Size([1])))
        print('support = ', dist.enumerate_support())
        node_samples_log_probs = dist.log_prob(node_samples)
        print('probs = ', node_samples_log_probs)