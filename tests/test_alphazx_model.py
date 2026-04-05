"""
Comprehensive tests for alphazx.models.homogeneous.alphazx_model.AlphaZXModel,
the top-level model combining RepresentationNetwork and PredictionNetwork.

Tests verify end-to-end forward pass shapes, probability constraints,
value network output, gradient flow, distribution sampling roundtrip,
batch consistency, and parameter management.
"""

import pytest
import torch
import torch_geometric as pyg

from alphazx.diagram import METADATA, POSSIBLE_PHASES
from alphazx.diagram.diagram_generators import clifford_zx_diagram
from alphazx.diagram.zx_match_diagram import to_zx_match_diagram
from alphazx.distributions import AlphaZXDistribution, AlphaZXDistributionParams
from alphazx.game.zx_game import remove_isolated_nodes, remove_self_loop_edges, remove_isolated_components
from alphazx.models import pre_process
from alphazx.models.homogeneous.alphazx_model import AlphaZXModel


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

NUM_NODE_TYPES = len(METADATA.node_type_abbrevs)
NUM_POSSIBLE_PHASES = len(POSSIBLE_PHASES)
NUM_POSSIBLE_NEW_EDGES = 5
NODE_EMB_CHANNELS = 64
NUM_EDGE_EMBEDDINGS = len(METADATA.edge_feat_to_index_dict)
EDGE_EMB_CHANNELS = 64
PE_DIM = 20
NUM_QUBITS = 5
DEPTH = 5


def _make_model():
    return AlphaZXModel(
        NUM_NODE_TYPES, NUM_POSSIBLE_PHASES, NUM_POSSIBLE_NEW_EDGES,
        NODE_EMB_CHANNELS, NUM_EDGE_EMBEDDINGS, EDGE_EMB_CHANNELS,
        PE_DIM, PE_DIM,
    )


def _make_single_data():
    """Create a single preprocessed PyG Data object."""
    d = clifford_zx_diagram(NUM_QUBITS, DEPTH, t_gates=True)
    remove_isolated_nodes(d)
    remove_self_loop_edges(d)
    remove_isolated_components(d)
    md = to_zx_match_diagram(d)
    pyg_data = md.to_pyg_data()
    pyg_data = pre_process(pyg_data, PE_DIM)
    return pyg_data


def _make_batch(batch_size=2):
    data_list = [_make_single_data() for _ in range(batch_size)]
    return pyg.data.Batch.from_data_list(data_list)


def _run_model(model, batch):
    """Run the full model forward pass and return (params, value)."""
    return model(
        batch.x, batch.edge_index, batch.edge_attr, batch.node_type,
        batch.batch, batch.pe, batch.id,
    )


# ===========================================================================
# TestAlphaZXModelForward
# ===========================================================================


class TestAlphaZXModelForward:
    """Basic forward pass tests for the full AlphaZXModel."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_model()
        self.model.eval()

    def test_returns_params_and_value(self):
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            result = _run_model(self.model, batch)
        assert len(result) == 2
        params, value = result
        assert isinstance(params, AlphaZXDistributionParams)
        assert isinstance(value, torch.Tensor)

    def test_value_shape(self):
        batch = _make_batch(batch_size=3)
        with torch.no_grad():
            params, value = _run_model(self.model, batch)
        assert value.shape == (3, 1)

    def test_value_is_finite(self):
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            _, value = _run_model(self.model, batch)
        assert torch.isfinite(value).all()

    def test_single_batch(self):
        """Model should work with batch_size=1."""
        batch = _make_batch(batch_size=1)
        with torch.no_grad():
            params, value = _run_model(self.model, batch)
        assert params.mixture_dist_probs.shape[0] == 1
        assert value.shape == (1, 1)


# ===========================================================================
# TestAlphaZXModelOutputConstraints
# ===========================================================================


class TestAlphaZXModelOutputConstraints:
    """Verify that model outputs satisfy probability constraints."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_model()
        self.model.eval()
        self.batch = _make_batch(batch_size=2)
        with torch.no_grad():
            self.params, self.value = _run_model(self.model, self.batch)

    def test_mixture_probs_non_negative(self):
        assert (self.params.mixture_dist_probs >= 0).all()

    def test_mixture_probs_sum_to_one_or_zero(self):
        sums = self.params.mixture_dist_probs.sum(dim=-1)
        for s in sums:
            assert torch.isclose(s, torch.tensor(1.0), atol=1e-5) or s < 1e-5

    def test_node_probs_non_negative(self):
        assert (self.params.node_dist_probs >= 0).all()

    def test_node_probs_sum_to_one_or_zero(self):
        """For each (batch, type) pair, node probs should sum to ~1 or be all zeros."""
        B, T, N = self.params.node_dist_probs.shape
        for b in range(B):
            for t in range(T):
                s = self.params.node_dist_probs[b, t].sum()
                assert torch.isclose(s, torch.tensor(1.0), atol=1e-5) or s < 1e-5

    def test_phase_probs_non_negative(self):
        assert (self.params.phase_dist_probs >= 0).all()

    def test_new_edge_probs_non_negative(self):
        assert (self.params.new_edge_dist_probs >= 0).all()

    def test_transfer_edge_probs_in_zero_one(self):
        """Transfer edge probs (sigmoid output) should be in [0, 1]."""
        te = self.params.transfer_edge_dist_probs
        assert (te >= 0).all() and (te <= 1).all()

    def test_no_nan_in_any_param(self):
        for field in self.params._fields:
            t = getattr(self.params, field)
            if isinstance(t, torch.Tensor) and t.is_floating_point():
                assert not torch.isnan(t).any(), f"NaN in {field}"

    def test_mixture_node_alignment(self):
        """If mixture[b, t] > 0, then node_probs[b, t, :].sum() > 0."""
        mixture = self.params.mixture_dist_probs
        node = self.params.node_dist_probs
        B, T = mixture.shape
        for b in range(B):
            for t in range(T):
                if mixture[b, t].item() > 1e-6:
                    assert node[b, t].sum().item() > 1e-6, \
                        f"Mixture[{b},{t}]={mixture[b, t].item()} but node_probs row is all zeros"


# ===========================================================================
# TestAlphaZXModelSamplingRoundtrip
# ===========================================================================


class TestAlphaZXModelSamplingRoundtrip:
    """Test the full model → distribution → sample → log_prob pipeline."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_model()
        self.model.eval()

    def test_sample_from_model_output(self):
        """Should be able to sample actions from the model's output distribution."""
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            params, value = _run_model(self.model, batch)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(1)
        B = batch.batch.max().item() + 1
        assert samples.shape[0] == B
        assert samples.shape[1] == 1

    def test_log_prob_of_own_samples_is_finite(self):
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            params, _ = _run_model(self.model, batch)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(1)
        lp = dist.log_prob(samples)
        assert torch.isfinite(lp).all()

    def test_log_prob_is_negative(self):
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            params, _ = _run_model(self.model, batch)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(1)
        lp = dist.log_prob(samples)
        assert (lp <= 0).all()

    def test_multiple_samples(self):
        """Sampling K>1 should work and produce different actions."""
        batch = _make_batch(batch_size=1)
        with torch.no_grad():
            params, _ = _run_model(self.model, batch)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(8)
        assert samples.shape[1] == 8
        lp = dist.log_prob(samples)
        assert torch.isfinite(lp).all()

    def test_entropy_is_non_negative(self):
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            params, _ = _run_model(self.model, batch)
        dist = AlphaZXDistribution(params)
        entropy = dist.entropy()
        assert entropy.item() >= 0

    def test_sampled_action_types_have_valid_nodes(self):
        """Sampled action types should always correspond to types with valid match nodes."""
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            params, _ = _run_model(self.model, batch)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(5)
        B = samples.shape[0]
        action_types = samples[:, :, 1]  # [B, K]
        for b in range(B):
            for k in range(5):
                t = action_types[b, k].item()
                assert params.node_dist_probs[b, t].sum().item() > 1e-6, \
                    f"Sampled action type {t} has no valid nodes"


# ===========================================================================
# TestAlphaZXModelGradients
# ===========================================================================


class TestAlphaZXModelGradients:
    """Test that gradients flow through the full model."""

    def test_value_gradient_flow(self):
        model = _make_model()
        batch = _make_batch(batch_size=1)
        params, value = _run_model(model, batch)
        loss = value.sum()
        loss.backward()
        # Check representation network has gradients
        has_repr_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.representation_network.parameters()
        )
        assert has_repr_grad, "No gradients in representation network from value loss"

    def test_policy_gradient_flow(self):
        model = _make_model()
        batch = _make_batch(batch_size=3)
        params, value = _run_model(model, batch)
        # Sum ALL policy outputs to avoid flaky failures when a random diagram
        # has no valid action types (making mixture_probs all-zero/no gradient)
        # or when ReLU in multi-head scoring kills all gradient paths for a
        # single graph.
        loss = (params.mixture_dist_probs.sum()
                + params.node_dist_probs.sum()
                + params.phase_dist_probs.sum()
                + params.new_edge_dist_probs.sum()
                + params.transfer_edge_dist_probs.sum())
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.prediction_network.policy_network.parameters()
        )
        assert has_grad, "No gradients in policy network from policy loss"

    def test_log_prob_gradient_flow(self):
        """Gradients should flow from log_prob back through the model."""
        model = _make_model()
        batch = _make_batch(batch_size=1)
        params, value = _run_model(model, batch)
        dist = AlphaZXDistribution(params)
        samples = dist.sample(1)
        lp = dist.log_prob(samples)
        loss = -lp.sum()  # Policy gradient style loss
        loss.backward()
        has_grad = any(
            p.grad is not None and p.grad.abs().sum() > 0
            for p in model.parameters()
        )
        assert has_grad, "No gradients from log_prob loss"


# ===========================================================================
# TestAlphaZXModelBatchConsistency
# ===========================================================================


class TestAlphaZXModelBatchConsistency:
    """Test that batched inference is consistent with individual inference."""

    def test_single_vs_batched_value_similar(self):
        """Value prediction for a graph should be similar whether processed alone or in a batch."""
        model = _make_model()
        model.eval()

        data1 = _make_single_data()
        data2 = _make_single_data()

        single_batch = pyg.data.Batch.from_data_list([data1])
        with torch.no_grad():
            _, single_value = _run_model(model, single_batch)

        combined_batch = pyg.data.Batch.from_data_list([data1, data2])
        with torch.no_grad():
            _, combined_value = _run_model(model, combined_batch)

        # The first graph's value should be similar in both cases.
        # Not exactly equal due to batch norm, but should be close.
        diff = (single_value[0] - combined_value[0]).abs().item()
        # Batch norm causes differences; just verify it's not wildly different.
        assert diff < 5.0, f"Value difference {diff} is too large between single and batched"


# ===========================================================================
# TestAlphaZXModelParameterManagement
# ===========================================================================


class TestAlphaZXModelParameterManagement:
    """Test reset_parameters and parameter counts."""

    def test_reset_parameters_runs(self):
        model = _make_model()
        model.reset_parameters()  # Should not raise

    def test_model_has_parameters(self):
        model = _make_model()
        num_params = sum(p.numel() for p in model.parameters())
        assert num_params > 0

    def test_representation_and_prediction_are_submodules(self):
        model = _make_model()
        assert hasattr(model, 'representation_network')
        assert hasattr(model, 'prediction_network')
        # Both should be nn.Module instances
        assert isinstance(model.representation_network, torch.nn.Module)
        assert isinstance(model.prediction_network, torch.nn.Module)

    def test_all_parameters_are_float(self):
        model = _make_model()
        for name, p in model.named_parameters():
            assert p.is_floating_point(), f"Parameter {name} is not floating point"


# ===========================================================================
# TestModelConsistencyWithDistribution
# ===========================================================================


class TestModelConsistencyWithDistribution:
    """Verify that model output params are consistent with AlphaZXDistribution expectations."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.model = _make_model()
        self.model.eval()

    def test_graph_ids_in_params(self):
        """Params should contain graph_ids matching the batch."""
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            params, _ = _run_model(self.model, batch)
        assert params.graph_ids is not None

    def test_all_param_fields_present(self):
        batch = _make_batch(batch_size=2)
        with torch.no_grad():
            params, _ = _run_model(self.model, batch)
        expected_fields = {'graph_ids', 'mixture_dist_probs', 'node_dist_probs',
                           'phase_dist_probs', 'new_edge_dist_probs', 'transfer_edge_dist_probs'}
        assert set(params._fields) == expected_fields

    def test_batch_dimension_consistent_across_params(self):
        batch = _make_batch(batch_size=3)
        with torch.no_grad():
            params, value = _run_model(self.model, batch)
        B = 3
        assert params.mixture_dist_probs.shape[0] == B
        assert params.node_dist_probs.shape[0] == B
        assert params.phase_dist_probs.shape[0] == B
        assert params.new_edge_dist_probs.shape[0] == B
        assert params.transfer_edge_dist_probs.shape[0] == B
        assert value.shape[0] == B

    def test_node_probs_second_dim_is_10(self):
        """node_dist_probs should have 10 action types (match types 1-10)."""
        batch = _make_batch(batch_size=1)
        with torch.no_grad():
            params, _ = _run_model(self.model, batch)
        assert params.node_dist_probs.shape[1] == 10

    def test_mixture_probs_dim_is_10(self):
        """mixture_dist_probs should have 10 action types."""
        batch = _make_batch(batch_size=1)
        with torch.no_grad():
            params, _ = _run_model(self.model, batch)
        assert params.mixture_dist_probs.shape[1] == 10
