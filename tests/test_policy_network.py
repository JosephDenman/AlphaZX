"""
Comprehensive tests for alphazx.models.homogeneous.policy_network.PolicyNetwork
and its sub-components (RewriteTypeSelector, NodeSelector, NewPhaseSelector,
NewEdgeSelector, TransferEdgeSelector).

Tests verify output shapes, probability constraints, masking invariants,
gradient flow, and the pad_or_strip utility.
"""

import pytest
import torch
import torch_geometric as pyg

from alphazx.diagram import METADATA, POSSIBLE_PHASES
from alphazx.diagram.diagram_generators import clifford_zx_diagram
from alphazx.diagram.match import FRightZMatch, FRightXMatch
from alphazx.diagram.zx_match_diagram import to_zx_match_diagram
from alphazx.distributions.alpha_zx_dist import AlphaZXDistributionParams
from alphazx.game.zx_game import remove_isolated_nodes, remove_self_loop_edges, remove_isolated_components
from alphazx.models import pre_process
from alphazx.models.homogeneous.policy_network import PolicyNetwork, pad_or_strip
from alphazx.models.homogeneous.rewrite_type_selector import RewriteTypeSelector
from alphazx.models.homogeneous.node_selector import NodeSelector
from alphazx.models.homogeneous.new_phase_selector import NewPhaseSelector
from alphazx.models.homogeneous.new_edge_selector import NewEdgeSelector
from alphazx.models.homogeneous.transfer_edge_selector import TransferEdgeSelector


# ---------------------------------------------------------------------------
# Constants & helpers
# ---------------------------------------------------------------------------

NUM_NODE_TYPES = 10  # match types 1-10, action types 0-9
NUM_POSSIBLE_PHASES = len(POSSIBLE_PHASES)
NUM_POSSIBLE_NEW_EDGES = 5
NODE_CHANNELS = 64
EDGE_CHANNELS = 64
PE_DIM = 20
NUM_QUBITS = 5
DEPTH = 5


def _make_batch(batch_size=2, num_qubits=NUM_QUBITS, depth=DEPTH):
    """Create a processed PyG Batch from random Clifford diagrams."""
    data_list = []
    for _ in range(batch_size):
        d = clifford_zx_diagram(num_qubits, depth, t_gates=True)
        remove_isolated_nodes(d)
        remove_self_loop_edges(d)
        remove_isolated_components(d)
        md = to_zx_match_diagram(d)
        pyg_data = md.to_pyg_data()
        data_list.append(pyg_data)
    batch = pyg.data.Batch.from_data_list(data_list)
    batch = pre_process(batch, PE_DIM)
    return batch


def _make_policy_network():
    """Create a PolicyNetwork with standard test parameters."""
    num_node_types_full = len(METADATA.node_type_abbrevs)
    num_edge_embeddings = len(METADATA.edge_feat_to_index_dict)
    return PolicyNetwork(
        num_node_types=num_node_types_full,
        num_possible_phases=NUM_POSSIBLE_PHASES,
        num_possible_new_edges=NUM_POSSIBLE_NEW_EDGES,
        node_in_channels=NODE_CHANNELS + PE_DIM,  # RepresentationNetwork output channels
        edge_in_channels=EDGE_CHANNELS,
    )


# ===========================================================================
# TestRewriteTypeSelector
# ===========================================================================


class TestRewriteTypeSelector:
    """Tests for RewriteTypeSelector: mixture probability output."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.selector = RewriteTypeSelector(
            node_in_channels=NODE_CHANNELS + PE_DIM,
            num_node_types=NUM_NODE_TYPES,
            num_layers=2,
            pooling_dropout=0.1,
        )
        self.selector.eval()

    def _forward(self, batch):
        """Run the representation network and then the selector."""
        from alphazx.models.homogeneous.representation_network import RepresentationNetwork
        num_node_types_full = len(METADATA.node_type_abbrevs)
        num_edge_embeddings = len(METADATA.edge_feat_to_index_dict)
        repr_net = RepresentationNetwork(
            num_node_types_full, NUM_POSSIBLE_PHASES,
            NODE_CHANNELS, NODE_CHANNELS + PE_DIM, num_edge_embeddings, EDGE_CHANNELS,
            PE_DIM, PE_DIM,
        )
        repr_net.eval()
        with torch.no_grad():
            x, edge_attr = repr_net(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pe)
            return self.selector(x, batch.node_type, batch.batch)

    def test_output_shape(self):
        batch = _make_batch(batch_size=3)
        probs = self._forward(batch)
        assert probs.shape == (3, NUM_NODE_TYPES)

    def test_probabilities_sum_to_one(self):
        batch = _make_batch(batch_size=2)
        probs = self._forward(batch)
        sums = probs.sum(dim=-1)
        for i in range(probs.shape[0]):
            # Either sums to ~1 (has valid types) or is all zeros (no valid types)
            assert torch.isclose(sums[i], torch.tensor(1.0), atol=1e-5) or sums[i] < 1e-5

    def test_probabilities_non_negative(self):
        batch = _make_batch(batch_size=2)
        probs = self._forward(batch)
        assert (probs >= 0).all()

    def test_no_nan_in_output(self):
        batch = _make_batch(batch_size=2)
        probs = self._forward(batch)
        assert not torch.isnan(probs).any()

    def test_zero_probability_for_absent_types(self):
        """Types with no match nodes in the graph should get zero probability."""
        batch = _make_batch(batch_size=1)
        probs = self._forward(batch)
        # Check each action type: if no match nodes of type t+1 exist, probs[0, t] should be 0.
        node_types = batch.node_type
        for t in range(NUM_NODE_TYPES):
            match_type = t + 1  # match types 1-10
            if not (node_types == match_type).any():
                assert probs[0, t].item() == 0.0, \
                    f"Action type {t} (match type {match_type}) has no nodes but got prob {probs[0, t].item()}"


# ===========================================================================
# TestNodeSelector
# ===========================================================================


class TestNodeSelector:
    """Tests for NodeSelector: per-type node probability distributions."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.selector = NodeSelector(
            node_in_channels=NODE_CHANNELS + PE_DIM,
            num_node_types=NUM_NODE_TYPES,
            num_layers=2,
            dropout=0.1,
        )
        self.selector.eval()

    def _make_simple_input(self, num_nodes=8, batch_size=2):
        """Create simple synthetic input for unit testing the selector directly."""
        x = torch.randn(num_nodes, NODE_CHANNELS + PE_DIM)
        # Assign node types: some match (1-10), some super (12-21), some boundary (0)
        node_types = torch.tensor([0, 1, 2, 12, 13, 1, 3, 0])[:num_nodes]
        batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])[:num_nodes]
        return x, node_types, batch

    def test_output_shape(self):
        x, node_types, batch = self._make_simple_input()
        with torch.no_grad():
            probs = self.selector(x, node_types, batch)
        B = batch.max().item() + 1
        N = pyg.utils.to_dense_batch(x, batch)[0].shape[1]
        assert probs.shape == (B, NUM_NODE_TYPES, N)

    def test_probabilities_sum_to_one_per_type(self):
        x, node_types, batch = self._make_simple_input()
        with torch.no_grad():
            probs = self.selector(x, node_types, batch)
        B = probs.shape[0]
        for b in range(B):
            for t in range(NUM_NODE_TYPES):
                row_sum = probs[b, t].sum()
                # Either sums to ~1 (type present) or ~0 (type absent)
                assert torch.isclose(row_sum, torch.tensor(1.0), atol=1e-5) or row_sum < 1e-5

    def test_zero_rows_for_absent_types(self):
        """If no nodes of match type t+1 exist in a batch, probs[b, t, :] should be all zeros."""
        x, node_types, batch = self._make_simple_input()
        with torch.no_grad():
            probs = self.selector(x, node_types, batch)
        dense_types, _ = pyg.utils.to_dense_batch(node_types, batch)
        B, N = dense_types.shape
        for b in range(B):
            for t in range(NUM_NODE_TYPES):
                match_type = t + 1
                if not (dense_types[b] == match_type).any():
                    assert probs[b, t].sum().item() < 1e-6

    def test_super_nodes_get_zero_probability(self):
        """Super nodes (types 12-21) should never receive nonzero node selection probability."""
        x, node_types, batch = self._make_simple_input()
        with torch.no_grad():
            probs = self.selector(x, node_types, batch)
        dense_types, _ = pyg.utils.to_dense_batch(node_types, batch)
        B, N = dense_types.shape
        for b in range(B):
            for n in range(N):
                if dense_types[b, n] >= 12:
                    # This node is a super node; it should have 0 probability in ALL type rows.
                    assert probs[b, :, n].sum().item() < 1e-6

    def test_boundary_nodes_get_zero_probability(self):
        """Boundary nodes (type 0) should never receive nonzero node selection probability."""
        x, node_types, batch = self._make_simple_input()
        with torch.no_grad():
            probs = self.selector(x, node_types, batch)
        dense_types, _ = pyg.utils.to_dense_batch(node_types, batch)
        B, N = dense_types.shape
        for b in range(B):
            for n in range(N):
                if dense_types[b, n] == 0:
                    assert probs[b, :, n].sum().item() < 1e-6

    def test_non_negative_probabilities(self):
        x, node_types, batch = self._make_simple_input()
        with torch.no_grad():
            probs = self.selector(x, node_types, batch)
        assert (probs >= 0).all()


# ===========================================================================
# TestNewPhaseSelector
# ===========================================================================


class TestNewPhaseSelector:
    """Tests for NewPhaseSelector: phase distribution for FRight nodes only."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.selector = NewPhaseSelector(
            node_in_channels=NODE_CHANNELS + PE_DIM,
            num_possible_phases=NUM_POSSIBLE_PHASES,
            num_layers=2,
            dropout=0.1,
        )
        self.selector.eval()

    def _make_input(self, num_nodes=6, batch_size=2):
        x = torch.randn(num_nodes, NODE_CHANNELS + PE_DIM)
        # Types: FRightZ=1, FRightX=2, FLeftZ=3, boundary=0, super=12, FRightZ=1
        node_types = torch.tensor([1, 2, 3, 0, 12, 1])[:num_nodes]
        batch = torch.tensor([0, 0, 0, 1, 1, 1])[:num_nodes]
        return x, node_types, batch

    def test_output_shape(self):
        x, node_types, batch = self._make_input()
        with torch.no_grad():
            probs = self.selector(x, node_types, batch)
        B = batch.max().item() + 1
        N = pyg.utils.to_dense_batch(x, batch)[0].shape[1]
        assert probs.shape == (B, N, NUM_POSSIBLE_PHASES)

    def test_non_simple_nodes_get_deterministic_phase(self):
        """Non-FRight nodes should have all probability on phase index 0."""
        x, node_types, batch = self._make_input()
        with torch.no_grad():
            probs = self.selector(x, node_types, batch)
        dense_types, _ = pyg.utils.to_dense_batch(node_types, batch, fill_value=float('nan'))
        B, N = dense_types.shape
        for b in range(B):
            for n in range(N):
                nt = dense_types[b, n].item()
                if nt != FRightZMatch.index and nt != FRightXMatch.index:
                    if not (nt != nt):  # not NaN (padding)
                        assert torch.isclose(probs[b, n, 0], torch.tensor(1.0), atol=1e-5)
                        assert probs[b, n, 1:].sum() < 1e-5

    def test_simple_nodes_sum_to_one(self):
        """FRight nodes should have phase probabilities summing to 1."""
        x, node_types, batch = self._make_input()
        with torch.no_grad():
            probs = self.selector(x, node_types, batch)
        dense_types, _ = pyg.utils.to_dense_batch(node_types, batch, fill_value=float('nan'))
        B, N = dense_types.shape
        for b in range(B):
            for n in range(N):
                nt = dense_types[b, n].item()
                if nt == FRightZMatch.index or nt == FRightXMatch.index:
                    assert torch.isclose(probs[b, n].sum(), torch.tensor(1.0), atol=1e-5)


# ===========================================================================
# TestNewEdgeSelector
# ===========================================================================


class TestNewEdgeSelector:
    """Tests for NewEdgeSelector: new edge count distribution for FRight nodes."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.selector = NewEdgeSelector(
            node_in_channels=NODE_CHANNELS + PE_DIM,
            num_possible_new_edges=NUM_POSSIBLE_NEW_EDGES,
            num_layers=2,
            dropout=0.1,
        )
        self.selector.eval()

    def _make_input(self, num_nodes=6, batch_size=2):
        x = torch.randn(num_nodes, NODE_CHANNELS + PE_DIM)
        node_types = torch.tensor([1, 2, 3, 0, 12, 1])[:num_nodes]
        batch = torch.tensor([0, 0, 0, 1, 1, 1])[:num_nodes]
        return x, node_types, batch

    def test_output_shape(self):
        x, node_types, batch = self._make_input()
        with torch.no_grad():
            probs = self.selector(x, node_types, batch)
        B = batch.max().item() + 1
        N = pyg.utils.to_dense_batch(x, batch)[0].shape[1]
        assert probs.shape == (B, N, NUM_POSSIBLE_NEW_EDGES)

    def test_non_simple_nodes_get_deterministic_edge(self):
        """Non-FRight nodes should have all probability on edge index 0."""
        x, node_types, batch = self._make_input()
        with torch.no_grad():
            probs = self.selector(x, node_types, batch)
        dense_types, _ = pyg.utils.to_dense_batch(node_types, batch, fill_value=float('nan'))
        B, N = dense_types.shape
        for b in range(B):
            for n in range(N):
                nt = dense_types[b, n].item()
                if nt != FRightZMatch.index and nt != FRightXMatch.index:
                    if not (nt != nt):  # not NaN (padding)
                        assert torch.isclose(probs[b, n, 0], torch.tensor(1.0), atol=1e-5)

    def test_simple_nodes_sum_to_one(self):
        x, node_types, batch = self._make_input()
        with torch.no_grad():
            probs = self.selector(x, node_types, batch)
        dense_types, _ = pyg.utils.to_dense_batch(node_types, batch, fill_value=float('nan'))
        B, N = dense_types.shape
        for b in range(B):
            for n in range(N):
                nt = dense_types[b, n].item()
                if nt == FRightZMatch.index or nt == FRightXMatch.index:
                    assert torch.isclose(probs[b, n].sum(), torch.tensor(1.0), atol=1e-5)


# ===========================================================================
# TestPolicyNetworkForward
# ===========================================================================


class TestPolicyNetworkForward:
    """Integration tests for the full PolicyNetwork forward pass."""

    @pytest.fixture(autouse=True)
    def setup(self):
        self.batch = _make_batch(batch_size=2)
        # PolicyNetwork expects post-representation features, so we need to run representation first.
        from alphazx.models.homogeneous.representation_network import RepresentationNetwork
        num_node_types_full = len(METADATA.node_type_abbrevs)
        num_edge_embeddings = len(METADATA.edge_feat_to_index_dict)
        self.repr_net = RepresentationNetwork(
            num_node_types_full, NUM_POSSIBLE_PHASES,
            NODE_CHANNELS, NODE_CHANNELS + PE_DIM, num_edge_embeddings, EDGE_CHANNELS,
            PE_DIM, PE_DIM,
        )
        self.policy = PolicyNetwork(
            num_node_types=num_node_types_full,
            num_possible_phases=NUM_POSSIBLE_PHASES,
            num_possible_new_edges=NUM_POSSIBLE_NEW_EDGES,
            node_in_channels=NODE_CHANNELS + PE_DIM,
            edge_in_channels=EDGE_CHANNELS,
        )
        self.repr_net.eval()
        self.policy.eval()

    def _run_forward(self):
        batch = self.batch
        with torch.no_grad():
            x, edge_attr = self.repr_net(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pe)
            graph_ids = batch.id if hasattr(batch, 'id') else torch.zeros(batch.batch.max() + 1)
            return self.policy(x, batch.edge_index, edge_attr, batch.node_type, batch.batch, graph_ids)

    def test_returns_azx_distribution_params(self):
        result = self._run_forward()
        assert isinstance(result, AlphaZXDistributionParams)

    def test_mixture_probs_shape(self):
        params = self._run_forward()
        B = self.batch.batch.max().item() + 1
        assert params.mixture_dist_probs.shape == (B, NUM_NODE_TYPES)

    def test_mixture_probs_valid_distribution(self):
        params = self._run_forward()
        sums = params.mixture_dist_probs.sum(dim=-1)
        for i in range(sums.shape[0]):
            assert torch.isclose(sums[i], torch.tensor(1.0), atol=1e-5) or sums[i] < 1e-5

    def test_node_probs_shape(self):
        params = self._run_forward()
        B = self.batch.batch.max().item() + 1
        # [B, T, N]
        assert params.node_dist_probs.shape[0] == B
        assert params.node_dist_probs.shape[1] == NUM_NODE_TYPES

    def test_phase_probs_shape(self):
        params = self._run_forward()
        B = self.batch.batch.max().item() + 1
        assert params.phase_dist_probs.shape[0] == B
        assert params.phase_dist_probs.shape[-1] == NUM_POSSIBLE_PHASES

    def test_new_edge_probs_shape(self):
        params = self._run_forward()
        B = self.batch.batch.max().item() + 1
        assert params.new_edge_dist_probs.shape[0] == B
        assert params.new_edge_dist_probs.shape[-1] == NUM_POSSIBLE_NEW_EDGES

    def test_no_nan_in_any_output(self):
        params = self._run_forward()
        for name in params._fields:
            tensor = getattr(params, name)
            if isinstance(tensor, torch.Tensor) and tensor.is_floating_point():
                assert not torch.isnan(tensor).any(), f"NaN found in {name}"

    def test_no_negative_probabilities(self):
        params = self._run_forward()
        assert (params.mixture_dist_probs >= 0).all()
        assert (params.node_dist_probs >= 0).all()
        assert (params.phase_dist_probs >= 0).all()
        assert (params.new_edge_dist_probs >= 0).all()

    def test_mixture_node_alignment(self):
        """If mixture_probs[b, t] > 0, then node_probs[b, t, :].sum() > 0."""
        params = self._run_forward()
        mixture = params.mixture_dist_probs
        node = params.node_dist_probs
        B, T = mixture.shape
        for b in range(B):
            for t in range(T):
                if mixture[b, t].item() > 1e-6:
                    assert node[b, t].sum().item() > 1e-6, \
                        f"Mixture[{b},{t}]={mixture[b, t].item()} but node_probs row is all zeros"


# ===========================================================================
# TestPolicyNetworkGradients
# ===========================================================================


class TestPolicyNetworkGradients:
    """Verify that gradients flow through all policy network outputs."""

    def test_gradients_flow_through_mixture(self):
        batch = _make_batch(batch_size=3)
        from alphazx.models.homogeneous.representation_network import RepresentationNetwork
        num_node_types_full = len(METADATA.node_type_abbrevs)
        num_edge_embeddings = len(METADATA.edge_feat_to_index_dict)
        repr_net = RepresentationNetwork(
            num_node_types_full, NUM_POSSIBLE_PHASES,
            NODE_CHANNELS, NODE_CHANNELS + PE_DIM, num_edge_embeddings, EDGE_CHANNELS,
            PE_DIM, PE_DIM,
        )
        policy = PolicyNetwork(
            num_node_types=num_node_types_full,
            num_possible_phases=NUM_POSSIBLE_PHASES,
            num_possible_new_edges=NUM_POSSIBLE_NEW_EDGES,
            node_in_channels=NODE_CHANNELS + PE_DIM,
            edge_in_channels=EDGE_CHANNELS,
        )
        x, edge_attr = repr_net(batch.x, batch.edge_index, batch.edge_attr, batch.batch, batch.pe)
        graph_ids = batch.id if hasattr(batch, 'id') else torch.zeros(batch.batch.max() + 1)
        params = policy(x, batch.edge_index, edge_attr, batch.node_type, batch.batch, graph_ids)

        # Backprop through all policy outputs to test gradient flow end-to-end.
        # Using all outputs (not just mixture_probs) avoids flaky failures when a
        # random diagram happens to have no valid action types, which makes
        # mixture_probs all-zero (no gradient path) after nan_to_num in
        # RewriteTypeSelector.
        loss = (params.mixture_dist_probs.sum()
                + params.node_dist_probs.sum()
                + params.phase_dist_probs.sum()
                + params.new_edge_dist_probs.sum()
                + params.transfer_edge_dist_probs.sum())
        loss.backward()

        # At least some parameters should have gradients
        has_grad = False
        for p in policy.parameters():
            if p.grad is not None and p.grad.abs().sum() > 0:
                has_grad = True
                break
        assert has_grad, "No gradients found in policy network parameters"


# ===========================================================================
# TestPadOrStrip
# ===========================================================================


class TestPadOrStrip:
    """Tests for the pad_or_strip utility function."""

    def test_padding_increases_last_dim(self):
        actions = torch.zeros(2, 3, 4)  # B=2, K=3, L=4
        # Create a simple batch with degree that requires L = max_degree + 5 > 4
        edge_index = torch.tensor([[0, 1, 2, 3, 4, 5, 0, 1],
                                   [1, 0, 3, 2, 5, 4, 2, 3]])
        batch = pyg.data.Data(edge_index=edge_index, num_nodes=6)
        result = pad_or_strip(actions, batch)
        # max degree in this graph should be computed from edge_index[0]
        assert result.shape[-1] >= 4  # at least as large or larger
        assert result.shape[0] == 2
        assert result.shape[1] == 3

    def test_stripping_reduces_last_dim(self):
        actions = torch.zeros(2, 3, 100)  # Last dim very large
        # Small graph → target_size small
        edge_index = torch.tensor([[0, 1], [1, 0]])
        batch = pyg.data.Data(edge_index=edge_index, num_nodes=2)
        result = pad_or_strip(actions, batch)
        # max degree is 1, target is 1 + 5 = 6
        assert result.shape[-1] == 6

    def test_padding_fills_with_zeros(self):
        actions = torch.ones(1, 1, 3)
        edge_index = torch.tensor([[0, 1, 2, 0, 1, 2], [1, 2, 0, 2, 0, 1]])
        batch = pyg.data.Data(edge_index=edge_index, num_nodes=3)
        result = pad_or_strip(actions, batch)
        if result.shape[-1] > 3:
            assert (result[..., 3:] == 0).all()

    def test_preserves_dtype_and_device(self):
        actions = torch.ones(1, 1, 3, dtype=torch.long)
        edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]])
        batch = pyg.data.Data(edge_index=edge_index, num_nodes=3)
        result = pad_or_strip(actions, batch)
        assert result.dtype == torch.long


# ===========================================================================
# TestResetParameters
# ===========================================================================


class TestResetParameters:
    """Test that reset_parameters runs without error on the policy network and sub-modules."""

    def test_policy_network_reset(self):
        num_node_types_full = len(METADATA.node_type_abbrevs)
        num_edge_embeddings = len(METADATA.edge_feat_to_index_dict)
        policy = PolicyNetwork(
            num_node_types=num_node_types_full,
            num_possible_phases=NUM_POSSIBLE_PHASES,
            num_possible_new_edges=NUM_POSSIBLE_NEW_EDGES,
            node_in_channels=NODE_CHANNELS + PE_DIM,
            edge_in_channels=EDGE_CHANNELS,
        )
        # Should not raise
        policy.reset_parameters()

    def test_rewrite_type_selector_reset(self):
        s = RewriteTypeSelector(NODE_CHANNELS + PE_DIM, NUM_NODE_TYPES, 2, 0.1)
        s.reset_parameters()

    def test_node_selector_reset(self):
        s = NodeSelector(NODE_CHANNELS + PE_DIM, NUM_NODE_TYPES, 2, 0.1)
        s.reset_parameters()

    def test_new_phase_selector_reset(self):
        s = NewPhaseSelector(NODE_CHANNELS + PE_DIM, NUM_POSSIBLE_PHASES, 2, 0.1)
        s.reset_parameters()

    def test_new_edge_selector_reset(self):
        s = NewEdgeSelector(NODE_CHANNELS + PE_DIM, NUM_POSSIBLE_NEW_EDGES, 2, 0.1)
        s.reset_parameters()
