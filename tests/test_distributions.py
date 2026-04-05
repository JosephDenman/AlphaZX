"""
Comprehensive tests for alphazx.distributions module.

Tests cover:
- MultivariateBernoulli: construction, sampling, log_prob, edge cases
- AlphaZXDistribution: sample/log_prob roundtrip, component decomposition,
  shape correctness, entropy, gradient flow
- Helper functions: safe_log, check_non_zero_elems_exist, check_non_zero_rows
"""

import math

import pytest
import torch

from alphazx.distributions.alpha_zx_dist import (
    AlphaZXDistribution,
    AlphaZXDistributionParams,
    safe_log,
    check_non_zero_elems_exist,
    check_non_zero_rows,
)
from alphazx.distributions.bernoulli_mixture import (
    MultivariateBernoulli,
    MultivariateBernoulliMixture,
)

torch.manual_seed(42)

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

NUM_ACTION_TYPES = 10  # Match types 1-10 → action types 0-9
NUM_PHASES = 8
NUM_NEW_EDGES = 5


def _make_dist_params(
    batch_size: int = 1,
    num_nodes: int = 6,
    max_degree: int = 4,
    num_action_types: int = NUM_ACTION_TYPES,
    num_phases: int = NUM_PHASES,
    num_new_edges: int = NUM_NEW_EDGES,
    require_grad: bool = False,
) -> AlphaZXDistributionParams:
    """Build valid AlphaZXDistributionParams with controllable sizes.

    Creates distributions where:
    - mixture_dist_probs: valid probability distribution over action types
    - node_dist_probs: for each action type, valid distribution over nodes
      (some types may have all-zero rows = no nodes of that type)
    - phase_dist_probs: per-node distribution over phases
    - new_edge_dist_probs: per-node distribution over new edge counts
    - transfer_edge_dist_probs: per-node Bernoulli params in [0, 1]
    """
    B, T, N, P, E_new, E_trans = (
        batch_size,
        num_action_types,
        num_nodes,
        num_phases,
        num_new_edges,
        max_degree,
    )

    # --- node probs [B, T, N] ---
    # For each (b, t), create a valid distribution (or all-zero if no nodes of that type).
    # Guarantee at least one type has valid nodes so actions can be sampled.
    node_raw = torch.rand(B, T, N) + 1e-4
    # Mask out some types entirely (set to zero) — but keep at least 2 types active
    active_mask = torch.zeros(B, T, dtype=torch.bool)
    for b in range(B):
        active_count = max(2, T // 2)
        active_indices = torch.randperm(T)[:active_count]
        active_mask[b, active_indices] = True
    node_raw[~active_mask.unsqueeze(-1).expand_as(node_raw)] = 0.0
    # Normalize active types
    node_sums = node_raw.sum(dim=-1, keepdim=True)
    node_probs = torch.where(node_sums > 0, node_raw / (node_sums + 1e-10), torch.zeros_like(node_raw))

    # --- mixture probs [B, T] ---
    # IMPORTANT: mixture must only assign probability to action types that have
    # valid node distributions (nonzero row in node_probs). Otherwise
    # Categorical(all_zeros).sample() crashes inside torch.multinomial.
    # This mirrors production behavior where RewriteTypeSelector masks out
    # action types with no match nodes.
    has_nodes = node_probs.sum(dim=-1) > 0  # [B, T]
    raw = (torch.rand(B, T) + 0.01) * has_nodes.float()
    mixture = raw / raw.sum(dim=-1, keepdim=True)

    # --- phase probs [B, T, N, P] (conditioned on action type) ---
    phase_raw = torch.rand(B, T, N, P) + 0.01
    phase_probs = phase_raw / phase_raw.sum(dim=-1, keepdim=True)

    # --- new edge probs [B, T, N, E_new] ---
    edge_raw = torch.rand(B, T, N, E_new) + 0.01
    new_edge_probs = edge_raw / edge_raw.sum(dim=-1, keepdim=True)

    # --- transfer edge probs [B, T, N, E_trans] --- in [0, 1]
    transfer_probs = torch.sigmoid(torch.randn(B, T, N, E_trans))

    # --- graph ids ---
    graph_ids = torch.arange(B)

    if require_grad:
        mixture = mixture.clone().requires_grad_(True)
        node_probs = node_probs.clone().requires_grad_(True)
        phase_probs = phase_probs.clone().requires_grad_(True)
        new_edge_probs = new_edge_probs.clone().requires_grad_(True)
        transfer_probs = transfer_probs.clone().requires_grad_(True)

    return AlphaZXDistributionParams(
        graph_ids=graph_ids,
        mixture_dist_probs=mixture,
        node_dist_probs=node_probs,
        phase_dist_probs=phase_probs,
        new_edge_dist_probs=new_edge_probs,
        transfer_edge_dist_probs=transfer_probs,
    )


# ===========================================================================
# MultivariateBernoulli tests
# ===========================================================================


class TestMultivariateBernoulli:
    """Tests for MultivariateBernoulli distribution wrapper."""

    def test_construction_valid_params(self):
        params = torch.tensor([[0.3, 0.7, 0.5]])
        dist = MultivariateBernoulli(params)
        assert dist.params.shape == (1, 3)

    def test_construction_rejects_negative(self):
        with pytest.raises(ValueError, match="range \\[0, 1\\]"):
            MultivariateBernoulli(torch.tensor([[-0.1, 0.5]]))

    def test_construction_rejects_above_one(self):
        with pytest.raises(ValueError, match="range \\[0, 1\\]"):
            MultivariateBernoulli(torch.tensor([[0.5, 1.1]]))

    def test_construction_boundary_values(self):
        """Exactly 0.0 and 1.0 should be accepted."""
        dist = MultivariateBernoulli(torch.tensor([[0.0, 1.0, 0.5]]))
        assert dist.params.shape == (1, 3)

    def test_sample_shape_no_sample_shape(self):
        params = torch.tensor([[0.5, 0.5, 0.5]])  # [1, 3]
        dist = MultivariateBernoulli(params)
        samples = dist.sample()
        assert samples.shape == (1, 3)

    def test_sample_shape_with_sample_shape(self):
        params = torch.tensor([[0.5, 0.5, 0.5]])  # [1, 3]
        dist = MultivariateBernoulli(params)
        samples = dist.sample(torch.Size([5]))
        assert samples.shape == (5, 1, 3)

    def test_sample_shape_batched(self):
        params = torch.rand(4, 6, 3)  # [B=4, N=6, E=3]
        dist = MultivariateBernoulli(params)
        samples = dist.sample(torch.Size([2]))
        assert samples.shape == (2, 4, 6, 3)

    def test_sample_values_binary(self):
        """Samples should be 0 or 1."""
        params = torch.rand(10, 5)
        dist = MultivariateBernoulli(params)
        samples = dist.sample(torch.Size([100]))
        assert torch.all((samples == 0) | (samples == 1))

    def test_sample_deterministic_zero(self):
        """Params = 0 should always produce 0."""
        params = torch.zeros(1, 5)
        dist = MultivariateBernoulli(params)
        samples = dist.sample(torch.Size([100]))
        assert torch.all(samples == 0)

    def test_sample_deterministic_one(self):
        """Params = 1 should always produce 1."""
        params = torch.ones(1, 5)
        dist = MultivariateBernoulli(params)
        samples = dist.sample(torch.Size([100]))
        assert torch.all(samples == 1)

    def test_log_prob_known_values(self):
        """Test log_prob against hand-computed values."""
        params = torch.tensor([[0.3, 0.7]])
        dist = MultivariateBernoulli(params)
        # P(0, 0) = (1-0.3) * (1-0.7) = 0.7 * 0.3 = 0.21
        lp_00 = dist.log_prob(torch.tensor([[0.0, 0.0]]))
        assert torch.isclose(lp_00, torch.tensor(math.log(0.21)), atol=1e-5)
        # P(1, 1) = 0.3 * 0.7 = 0.21
        lp_11 = dist.log_prob(torch.tensor([[1.0, 1.0]]))
        assert torch.isclose(lp_11, torch.tensor(math.log(0.21)), atol=1e-5)
        # P(1, 0) = 0.3 * 0.3 = 0.09
        lp_10 = dist.log_prob(torch.tensor([[1.0, 0.0]]))
        assert torch.isclose(lp_10, torch.tensor(math.log(0.09)), atol=1e-5)
        # P(0, 1) = 0.7 * 0.7 = 0.49
        lp_01 = dist.log_prob(torch.tensor([[0.0, 1.0]]))
        assert torch.isclose(lp_01, torch.tensor(math.log(0.49)), atol=1e-5)

    def test_log_prob_all_outcomes_sum_to_one(self):
        """For a 2D Bernoulli, exp(log_prob) over all 4 outcomes should sum to 1."""
        params = torch.tensor([[0.4, 0.8]])
        dist = MultivariateBernoulli(params)
        outcomes = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
        # Need to expand params for broadcasting
        dist_expanded = MultivariateBernoulli(params.expand(4, -1))
        log_probs = dist_expanded.log_prob(outcomes)
        total = log_probs.exp().sum()
        assert torch.isclose(total, torch.tensor(1.0), atol=1e-5)

    def test_log_prob_deterministic_param_zero(self):
        """log P(x=1) should be extremely negative when param=0."""
        params = torch.tensor([[0.0]])
        dist = MultivariateBernoulli(params)
        lp = dist.log_prob(torch.tensor([[1.0]]))
        # PyTorch internally clamps probabilities, so we get a very large
        # negative number (~-15.94) rather than exact -inf.
        assert lp.item() < -10

    def test_log_prob_deterministic_param_one(self):
        """log P(x=0) should be extremely negative when param=1."""
        params = torch.tensor([[1.0]])
        dist = MultivariateBernoulli(params)
        lp = dist.log_prob(torch.tensor([[0.0]]))
        # PyTorch internally clamps probabilities, so we get a very large
        # negative number (~-15.94) rather than exact -inf.
        assert lp.item() < -10

    def test_log_prob_shape_batched(self):
        params = torch.rand(3, 5, 4)  # [3, 5, 4]
        dist = MultivariateBernoulli(params)
        values = torch.randint(0, 2, (3, 5, 4)).float()
        lp = dist.log_prob(values)
        assert lp.shape == (3, 5)  # Independent reduces last dim

    def test_sample_statistical_mean(self):
        """Sample mean should approximate params for large sample size."""
        params = torch.tensor([[0.2, 0.8, 0.5]])
        dist = MultivariateBernoulli(params)
        samples = dist.sample(torch.Size([10000]))  # [10000, 1, 3]
        sample_mean = samples.float().mean(dim=0)
        assert torch.allclose(sample_mean, params.float(), atol=0.05)


# ===========================================================================
# MultivariateBernoulliMixture tests
# ===========================================================================


class TestMultivariateBernoulliMixture:
    """Tests for MultivariateBernoulliMixture distribution."""

    def _make_valid_params(self, B=1, N=1, K=3, E=2):
        """Create valid mixture params of shape [B, N, K, E+1].

        First column = mixture coefficients (sum to 1 across K).
        Remaining columns = Bernoulli params in [0, 1].
        """
        params = torch.rand(B, N, K, E + 1)
        # Normalize mixture coefficients (first column) to sum to 1 across K
        mixture_col = params[..., 0]
        mixture_col = mixture_col / mixture_col.sum(dim=-1, keepdim=True)
        params[..., 0] = mixture_col
        return params

    def test_construction_valid(self):
        params = self._make_valid_params()
        dist = MultivariateBernoulliMixture(params)
        assert dist.dist is not None

    def test_construction_rejects_bad_mixture_sum(self):
        params = self._make_valid_params()
        params[..., 0] = 0.5  # Won't sum to 1 unless K=2
        with pytest.raises(ValueError):
            MultivariateBernoulliMixture(params)

    def test_sample_shape(self):
        params = self._make_valid_params(B=2, N=3, K=4, E=5)
        dist = MultivariateBernoulliMixture(params)
        samples = dist.sample(torch.Size([7]))
        # MixtureSameFamily batch_shape = [B, N], event_shape = [E]
        assert samples.shape == (7, 2, 3, 5)

    def test_log_prob_finite(self):
        params = self._make_valid_params(B=2, N=3, K=2, E=3)
        dist = MultivariateBernoulliMixture(params)
        values = torch.randint(0, 2, (2, 3, 3)).float()
        lp = dist.log_prob(values)
        assert lp.shape == (2, 3)
        assert torch.all(torch.isfinite(lp))


# ===========================================================================
# AlphaZXDistribution tests
# ===========================================================================


class TestAlphaZXDistributionSample:
    """Tests for AlphaZXDistribution.sample() shape and value correctness."""

    def test_sample_shape_single_batch(self):
        params = _make_dist_params(batch_size=1, num_nodes=8, max_degree=3)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(5)
        # Expected shape: (B, K, 1 + 4 + E_trans) = (1, 5, 1 + 4 + 3)
        assert samples.shape == (1, 5, 1 + 4 + 3)

    def test_sample_shape_multi_batch(self):
        params = _make_dist_params(batch_size=4, num_nodes=6, max_degree=5)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(3)
        assert samples.shape == (4, 3, 1 + 4 + 5)

    def test_sample_graph_ids_correct(self):
        """Graph IDs in samples should match the distribution's graph_ids."""
        params = _make_dist_params(batch_size=3)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(10)
        for b in range(3):
            assert torch.all(samples[b, :, 0] == b)

    def test_sample_action_types_in_range(self):
        params = _make_dist_params(batch_size=2, num_action_types=NUM_ACTION_TYPES)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(50)
        action_types = samples[:, :, 1]
        assert torch.all(action_types >= 0)
        assert torch.all(action_types < NUM_ACTION_TYPES)

    def test_sample_node_indices_in_range(self):
        N = 8
        params = _make_dist_params(batch_size=1, num_nodes=N)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(50)
        nodes = samples[:, :, 2]
        assert torch.all(nodes >= 0)
        assert torch.all(nodes < N)

    def test_sample_phase_indices_in_range(self):
        P = NUM_PHASES
        params = _make_dist_params(batch_size=1)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(50)
        phases = samples[:, :, 3]
        assert torch.all(phases >= 0)
        assert torch.all(phases < P)

    def test_sample_new_edge_indices_in_range(self):
        E = NUM_NEW_EDGES
        params = _make_dist_params(batch_size=1)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(50)
        new_edges = samples[:, :, 4]
        assert torch.all(new_edges >= 0)
        assert torch.all(new_edges < E)

    def test_sample_transfer_edges_binary(self):
        """Transfer edges should be 0 or 1 (from .long() on Bernoulli samples)."""
        params = _make_dist_params(batch_size=1, max_degree=5)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(50)
        transfer = samples[:, :, 5:]
        assert torch.all((transfer == 0) | (transfer == 1))

    def test_sample_is_long_tensor(self):
        params = _make_dist_params()
        dist = AlphaZXDistribution(params)
        samples = dist.sample(3)
        assert samples.dtype == torch.long

    def test_sample_single_sample(self):
        """k=1 should work without errors."""
        params = _make_dist_params(batch_size=1)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(1)
        assert samples.shape[1] == 1


class TestAlphaZXDistributionLogProb:
    """Tests for AlphaZXDistribution.log_prob() correctness."""

    def test_log_prob_on_own_samples_is_finite(self):
        """log_prob of actions sampled from the distribution should be finite."""
        params = _make_dist_params(batch_size=2, num_nodes=6)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(10)
        lp = dist.log_prob(samples)
        assert lp.shape == (2, 10)
        assert torch.all(torch.isfinite(lp))

    def test_log_prob_is_negative(self):
        """log probabilities should be ≤ 0."""
        params = _make_dist_params(batch_size=1, num_nodes=5)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(20)
        lp = dist.log_prob(samples)
        assert torch.all(lp <= 1e-5)  # Small tolerance for numerical issues

    def test_log_prob_matches_component_sum(self):
        """Total log_prob should equal sum of component log probs.

        This is the key factorization property:
        log P(a) = log P(type) + log P(node|type) + log P(phase|node)
                   + log P(new_edges|node) + log P(transfer_edges|node)
        """
        params = _make_dist_params(batch_size=1, num_nodes=6, max_degree=3)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(5)

        # Full log_prob
        total_lp = dist.log_prob(samples)

        # Component-wise
        action_types = samples[:, :, 1]
        nodes = samples[:, :, 2]
        phases = samples[:, :, 3]
        new_edges = samples[:, :, 4]
        transfer_edges = samples[:, :, 5:]

        lp_type = dist.action_type_log_probs(action_types)
        lp_node = dist.node_log_probs(action_types, nodes)
        lp_phase = dist.new_phase_log_probs(action_types, nodes, phases)
        lp_edge = dist.new_edge_log_probs(action_types, nodes, new_edges)
        lp_transfer = dist.transfer_edge_log_probs(action_types, nodes, transfer_edges)

        component_sum = lp_type + lp_node + lp_phase + lp_edge + lp_transfer
        assert torch.allclose(total_lp, component_sum, atol=1e-5)

    def test_log_prob_graph_id_mismatch_raises(self):
        """Passing wrong graph_ids should raise an exception."""
        params = _make_dist_params(batch_size=1)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(3)
        # Corrupt graph_id
        samples[:, :, 0] = 999
        with pytest.raises(Exception, match="Expected graph ids"):
            dist.log_prob(samples)

    def test_log_prob_deterministic_check(self):
        """For a point distribution (single possible action), log_prob should be ~0."""
        B, T, N = 1, NUM_ACTION_TYPES, 3
        # Mixture: all mass on action type 0
        mixture = torch.zeros(B, T)
        mixture[0, 0] = 1.0
        # Node: all mass on node 0 for type 0
        node_probs = torch.zeros(B, T, N)
        node_probs[0, 0, 0] = 1.0
        # Phase: all mass on phase 0 (now [B, T, N, P])
        phase_probs = torch.zeros(B, T, N, NUM_PHASES)
        phase_probs[:, :, :, 0] = 1.0
        # New edges: all mass on 0 (now [B, T, N, E_new])
        new_edge_probs = torch.zeros(B, T, N, NUM_NEW_EDGES)
        new_edge_probs[:, :, :, 0] = 1.0
        # Transfer: all zeros (deterministic) (now [B, T, N, E_trans])
        transfer_probs = torch.zeros(B, T, N, 4)

        params = AlphaZXDistributionParams(
            graph_ids=torch.tensor([0]),
            mixture_dist_probs=mixture,
            node_dist_probs=node_probs,
            phase_dist_probs=phase_probs,
            new_edge_dist_probs=new_edge_probs,
            transfer_edge_dist_probs=transfer_probs,
        )
        dist = AlphaZXDistribution(params)

        # The only valid action: type=0, node=0, phase=0, new_edges=0, transfer=[0,0,0,0]
        action = torch.tensor([[[0, 0, 0, 0, 0, 0, 0, 0, 0]]])  # [B=1, K=1, L=9]
        lp = dist.log_prob(action)
        # Should be log(1.0) = 0.0 for the categorical components
        # Transfer edge log_prob: log(1) * 4 = 0  (P(0) when param=0 is 1)
        assert torch.isclose(lp, torch.tensor([[0.0]]), atol=1e-5)


class TestAlphaZXDistributionRoundtrip:
    """Sample-then-log_prob roundtrip consistency tests."""

    @pytest.mark.parametrize("batch_size", [1, 3])
    @pytest.mark.parametrize("num_nodes", [4, 10])
    def test_roundtrip_finite(self, batch_size, num_nodes):
        """Sample → log_prob should always produce finite values."""
        params = _make_dist_params(batch_size=batch_size, num_nodes=num_nodes)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(8)
        lp = dist.log_prob(samples)
        assert torch.all(torch.isfinite(lp)), f"Non-finite log_probs: {lp}"

    def test_different_samples_different_log_probs(self):
        """Different samples should generally have different log probabilities."""
        params = _make_dist_params(batch_size=1, num_nodes=10)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(20)
        lp = dist.log_prob(samples)
        # With 20 samples from a non-trivial distribution, not all should be identical
        unique_lp = torch.unique(lp)
        assert unique_lp.numel() > 1


class TestAlphaZXDistributionComponentLogProbs:
    """Tests for individual component log_prob methods."""

    def test_action_type_log_probs_shape(self):
        params = _make_dist_params(batch_size=2)
        dist = AlphaZXDistribution(params)
        action_types = torch.tensor([[0, 3], [1, 5]])  # (B=2, K=2)
        lp = dist.action_type_log_probs(action_types)
        assert lp.shape == (2, 2)

    def test_action_type_log_probs_sum_to_one(self):
        """exp(log_prob) over all types should sum to ~1 for each batch."""
        params = _make_dist_params(batch_size=1)
        dist = AlphaZXDistribution(params)
        all_types = torch.arange(NUM_ACTION_TYPES).unsqueeze(0)  # (1, T)
        lp = dist.action_type_log_probs(all_types)
        total = lp.exp().sum(dim=-1)
        assert torch.isclose(total, torch.tensor([1.0]), atol=1e-5)

    def test_node_log_probs_valid_for_active_types(self):
        """For action types with valid nodes, node log probs should be finite."""
        params = _make_dist_params(batch_size=1, num_nodes=6)
        dist = AlphaZXDistribution(params)
        # Find an active type (nonzero node distribution)
        for t in range(NUM_ACTION_TYPES):
            node_params = dist.node_dist_params[0, t, :]
            if node_params.sum() > 0:
                action_types = torch.tensor([[t]])
                # Pick a node with nonzero probability
                valid_nodes = (node_params > 0).nonzero(as_tuple=True)[0]
                node = valid_nodes[0].unsqueeze(0).unsqueeze(0)
                lp = dist.node_log_probs(action_types, node)
                assert torch.all(torch.isfinite(lp))
                return
        pytest.skip("No active types in random params")

    def test_node_log_probs_zero_prob_returns_neg_inf(self):
        """Selecting a node with 0 probability should return -inf."""
        B, T, N = 1, NUM_ACTION_TYPES, 5
        node_probs = torch.zeros(B, T, N)
        node_probs[0, 0, 0] = 1.0  # Only node 0 has probability for type 0

        params = _make_dist_params(batch_size=B, num_nodes=N)
        # Override node probs
        params = params._replace(node_dist_probs=node_probs)
        dist = AlphaZXDistribution(params)

        # Request log_prob for node 2 under type 0 (has 0 probability)
        action_types = torch.tensor([[0]])
        nodes = torch.tensor([[2]])
        lp = dist.node_log_probs(action_types, nodes)
        # PyTorch's Categorical internally clamps probabilities, yielding a
        # very large negative number (~-15.94) rather than exact -inf.
        assert lp.item() < -10

    def test_transfer_edge_log_probs_shape(self):
        E = 4
        params = _make_dist_params(batch_size=2, num_nodes=5, max_degree=E)
        dist = AlphaZXDistribution(params)
        action_types = torch.tensor([[0, 1], [2, 0]])  # (B=2, K=2)
        nodes = torch.tensor([[0, 1], [2, 3]])
        transfer = torch.randint(0, 2, (2, 2, E)).float()
        lp = dist.transfer_edge_log_probs(action_types, nodes, transfer)
        assert lp.shape == (2, 2)

    def test_phase_log_probs_shape(self):
        params = _make_dist_params(batch_size=2, num_nodes=5)
        dist = AlphaZXDistribution(params)
        action_types = torch.tensor([[0, 1], [2, 0]])
        nodes = torch.tensor([[0, 1], [2, 3]])
        phases = torch.tensor([[0, 2], [1, 0]])
        lp = dist.new_phase_log_probs(action_types, nodes, phases)
        assert lp.shape == (2, 2)


class TestAlphaZXDistributionEntropy:
    """Tests for AlphaZXDistribution.entropy() method."""

    def test_entropy_is_non_negative(self):
        params = _make_dist_params(batch_size=2, num_nodes=6)
        dist = AlphaZXDistribution(params)
        ent = dist.entropy()
        assert ent.item() >= 0

    def test_entropy_scalar(self):
        params = _make_dist_params(batch_size=1)
        dist = AlphaZXDistribution(params)
        ent = dist.entropy()
        assert ent.dim() == 0  # scalar

    def test_entropy_point_distribution_is_zero(self):
        """A fully deterministic distribution should have entropy ~0."""
        B, T, N = 1, NUM_ACTION_TYPES, 3
        mixture = torch.zeros(B, T)
        mixture[0, 0] = 1.0
        node_probs = torch.zeros(B, T, N)
        node_probs[0, 0, 0] = 1.0
        phase_probs = torch.zeros(B, T, N, NUM_PHASES)
        phase_probs[:, :, :, 0] = 1.0
        new_edge_probs = torch.zeros(B, T, N, NUM_NEW_EDGES)
        new_edge_probs[:, :, :, 0] = 1.0
        transfer_probs = torch.zeros(B, T, N, 2)

        params = AlphaZXDistributionParams(
            graph_ids=torch.tensor([0]),
            mixture_dist_probs=mixture,
            node_dist_probs=node_probs,
            phase_dist_probs=phase_probs,
            new_edge_dist_probs=new_edge_probs,
            transfer_edge_dist_probs=transfer_probs,
        )
        dist = AlphaZXDistribution(params)
        ent = dist.entropy()
        assert ent.item() < 0.1  # Should be very close to 0

    def test_entropy_uniform_is_higher(self):
        """A more uniform distribution should have higher entropy."""
        # Nearly uniform
        B, T, N = 1, NUM_ACTION_TYPES, 5
        mixture_uniform = torch.ones(B, T) / T
        node_uniform = torch.ones(B, T, N) / N
        phase_probs = torch.ones(B, T, N, NUM_PHASES) / NUM_PHASES
        edge_probs = torch.ones(B, T, N, NUM_NEW_EDGES) / NUM_NEW_EDGES
        transfer_probs = torch.full((B, T, N, 3), 0.5)

        params_uniform = AlphaZXDistributionParams(
            graph_ids=torch.tensor([0]),
            mixture_dist_probs=mixture_uniform,
            node_dist_probs=node_uniform,
            phase_dist_probs=phase_probs,
            new_edge_dist_probs=edge_probs,
            transfer_edge_dist_probs=transfer_probs,
        )

        # Concentrated
        mixture_conc = torch.zeros(B, T)
        mixture_conc[0, 0] = 1.0
        node_conc = torch.zeros(B, T, N)
        node_conc[0, 0, 0] = 1.0

        params_conc = AlphaZXDistributionParams(
            graph_ids=torch.tensor([0]),
            mixture_dist_probs=mixture_conc,
            node_dist_probs=node_conc,
            phase_dist_probs=phase_probs,
            new_edge_dist_probs=edge_probs,
            transfer_edge_dist_probs=transfer_probs,
        )

        dist_uniform = AlphaZXDistribution(params_uniform)
        dist_conc = AlphaZXDistribution(params_conc)

        assert dist_uniform.entropy().item() > dist_conc.entropy().item()


# ===========================================================================
# Gradient flow tests
# ===========================================================================


class TestGradientFlow:
    """Test that log_prob supports gradient computation for training."""

    def test_log_prob_gradients_flow_to_mixture(self):
        """Gradients should flow back through mixture_dist_probs."""
        params = _make_dist_params(batch_size=1, num_nodes=5, require_grad=True)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(3)
        lp = dist.log_prob(samples)
        loss = -lp.mean()
        loss.backward()
        assert params.mixture_dist_probs.grad is not None
        assert torch.any(params.mixture_dist_probs.grad != 0)

    def test_log_prob_gradients_flow_to_node_probs(self):
        params = _make_dist_params(batch_size=1, num_nodes=5, require_grad=True)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(3)
        lp = dist.log_prob(samples)
        loss = -lp.mean()
        loss.backward()
        assert params.node_dist_probs.grad is not None

    def test_log_prob_gradients_flow_to_phase_probs(self):
        params = _make_dist_params(batch_size=1, num_nodes=5, require_grad=True)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(3)
        lp = dist.log_prob(samples)
        loss = -lp.mean()
        loss.backward()
        assert params.phase_dist_probs.grad is not None

    def test_log_prob_gradients_flow_to_transfer_probs(self):
        params = _make_dist_params(batch_size=1, num_nodes=5, require_grad=True)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(3)
        lp = dist.log_prob(samples)
        loss = -lp.mean()
        loss.backward()
        assert params.transfer_edge_dist_probs.grad is not None

    def test_component_log_prob_gradients(self):
        """Individual component log_probs should also be differentiable."""
        params = _make_dist_params(batch_size=1, num_nodes=5, require_grad=True)
        dist = AlphaZXDistribution(params)

        action_types = torch.tensor([[0]])
        nodes = torch.tensor([[0]])
        phases = torch.tensor([[0]])

        lp = dist.action_type_log_probs(action_types)
        lp.backward()
        assert params.mixture_dist_probs.grad is not None


# ===========================================================================
# Helper function tests
# ===========================================================================


class TestSafeLog:
    def test_safe_log_positive_value(self):
        t = torch.tensor([1.0, 2.0, 0.5])
        result = safe_log(t)
        expected = torch.log(t)
        assert torch.allclose(result, expected)

    def test_safe_log_zero_is_finite(self):
        """safe_log(0) should not be -inf (it clamps to eps first)."""
        t = torch.tensor([0.0])
        result = safe_log(t)
        assert torch.isfinite(result)

    def test_safe_log_very_small_is_finite(self):
        t = torch.tensor([1e-45])
        result = safe_log(t)
        assert torch.isfinite(result)

    def test_safe_log_negative_is_clamped(self):
        """Negative values get clamped to eps, so safe_log is finite."""
        t = torch.tensor([-1.0])
        result = safe_log(t)
        assert torch.isfinite(result)


class TestCheckNonZeroElemsExist:
    def test_passes_for_nonzero_tensor(self):
        check_non_zero_elems_exist(torch.tensor([0.0, 1.0, 0.0]))

    def test_fails_for_all_zero(self):
        with pytest.raises(AssertionError):
            check_non_zero_elems_exist(torch.zeros(5))


class TestCheckNonZeroRows:
    def test_passes_for_valid_rows(self):
        t = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        check_non_zero_rows(t)

    def test_fails_for_zero_row(self):
        t = torch.tensor([[1.0, 0.0], [0.0, 0.0]])
        with pytest.raises(Exception):
            check_non_zero_rows(t)


# ===========================================================================
# AlphaZXDistributionParams construction tests
# ===========================================================================


class TestAlphaZXDistributionParams:
    def test_is_named_tuple(self):
        params = _make_dist_params()
        assert hasattr(params, "_fields")
        assert "graph_ids" in params._fields
        assert "mixture_dist_probs" in params._fields
        assert "node_dist_probs" in params._fields
        assert "phase_dist_probs" in params._fields
        assert "new_edge_dist_probs" in params._fields
        assert "transfer_edge_dist_probs" in params._fields

    def test_field_count(self):
        params = _make_dist_params()
        assert len(params._fields) == 6

    def test_replace_preserves_other_fields(self):
        params = _make_dist_params(batch_size=2)
        new_mixture = torch.ones(2, NUM_ACTION_TYPES) / NUM_ACTION_TYPES
        replaced = params._replace(mixture_dist_probs=new_mixture)
        assert torch.equal(replaced.node_dist_probs, params.node_dist_probs)
        assert torch.equal(replaced.mixture_dist_probs, new_mixture)


# ===========================================================================
# Edge case and stress tests
# ===========================================================================


class TestEdgeCases:
    def test_single_node_graph(self):
        """Distribution with a single node should work."""
        params = _make_dist_params(batch_size=1, num_nodes=1, max_degree=1)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(5)
        assert samples.shape[1] == 5
        lp = dist.log_prob(samples)
        assert torch.all(torch.isfinite(lp))

    def test_large_batch(self):
        """Larger batch sizes should work correctly."""
        params = _make_dist_params(batch_size=8, num_nodes=10, max_degree=6)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(4)
        assert samples.shape == (8, 4, 1 + 4 + 6)
        lp = dist.log_prob(samples)
        assert lp.shape == (8, 4)

    def test_many_samples(self):
        """Large number of samples should work."""
        params = _make_dist_params(batch_size=1, num_nodes=5)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(100)
        assert samples.shape[1] == 100

    def test_all_mass_on_one_action_type(self):
        """When mixture is degenerate, all samples should have the same action type."""
        params = _make_dist_params(batch_size=1, num_nodes=5)
        # Put all mass on type 3
        mixture = torch.zeros(1, NUM_ACTION_TYPES)
        mixture[0, 3] = 1.0
        # Must have valid nodes for type 3
        node_probs = params.node_dist_probs.clone()
        node_probs[0, 3, :] = 1.0 / params.node_dist_probs.shape[2]
        params = params._replace(mixture_dist_probs=mixture, node_dist_probs=node_probs)

        dist = AlphaZXDistribution(params)
        samples = dist.sample(20)
        action_types = samples[0, :, 1]
        assert torch.all(action_types == 3)

    def test_probs_method(self):
        """The probs() method should return exp(log_prob)."""
        params = _make_dist_params(batch_size=1, num_nodes=5)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(5)
        probs = dist.probs(samples)
        lp = dist.log_prob(samples)
        expected = lp.exp()
        assert torch.allclose(probs, expected, atol=1e-6)

    def test_zero_transfer_edge_params(self):
        """Transfer probs all zero should produce all-zero transfer edges."""
        params = _make_dist_params(batch_size=1, num_nodes=3, max_degree=4)
        T = params.transfer_edge_dist_probs.shape[1]
        transfer = torch.zeros(1, T, 3, 4)
        params = params._replace(transfer_edge_dist_probs=transfer)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(10)
        transfer_samples = samples[:, :, 5:]
        assert torch.all(transfer_samples == 0)

    def test_one_transfer_edge_params(self):
        """Transfer probs all 1.0 should produce all-one transfer edges."""
        params = _make_dist_params(batch_size=1, num_nodes=3, max_degree=4)
        T = params.transfer_edge_dist_probs.shape[1]
        transfer = torch.ones(1, T, 3, 4)
        params = params._replace(transfer_edge_dist_probs=transfer)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(10)
        transfer_samples = samples[:, :, 5:]
        assert torch.all(transfer_samples == 1)
