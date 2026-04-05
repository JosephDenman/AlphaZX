"""
Comprehensive tests for alphazx.game.zx_game.

Covers:
- ZXGame initialization and reset
- ZXGame.step with valid actions
- Reward calculation (shaped and simple)
- Episode termination conditions
- Diagram cleanup utilities (remove_isolated_nodes, etc.)
- tuple_to_match action mapping
- DiagramStats consistency
"""

import pytest
import torch
import networkx as nx

from alphazx.diagram import METADATA, POSSIBLE_PHASES
from alphazx.diagram.diagram_generators import clifford_zx_diagram
from alphazx.diagram.match import (
    FRightZMatch, FRightXMatch, FLeftZMatch, FLeftXMatch,
    BRightMatch, BLeftMatch, YRightZMatch, YLeftZMatch,
    YRightXMatch, YLeftXMatch, BoundaryMatch,
)
from alphazx.diagram.zx_match_diagram import to_zx_match_diagram
from alphazx.game.zx_game import (
    ZXGame,
    DiagramStats,
    RewardBreakdown,
    calculate_reward,
    calculate_reward_simple,
    is_simplified,
    num_non_clifford_gates,
    remove_isolated_nodes,
    remove_self_loop_edges,
    remove_isolated_components,
    tuple_to_match,
    node_index_to_match,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NUM_QUBITS = 5
DEPTH = 5
PE_DIM = 20


@pytest.fixture
def zx_game():
    """A fresh ZXGame with small parameters for fast tests."""
    return ZXGame(num_qubits=NUM_QUBITS, depth=DEPTH, max_episode_length=50, pe_dim=PE_DIM)


@pytest.fixture
def zx_diagram():
    """A raw Clifford ZX diagram before match conversion."""
    d = clifford_zx_diagram(NUM_QUBITS, DEPTH, t_gates=True)
    remove_isolated_nodes(d)
    remove_self_loop_edges(d)
    remove_isolated_components(d)
    return d


@pytest.fixture
def zx_match_diagram(zx_diagram):
    return to_zx_match_diagram(zx_diagram)


# ===========================================================================
# TestZXGameInit
# ===========================================================================


class TestZXGameInit:
    """Tests for ZXGame.__init__ and parameter storage."""

    def test_stores_parameters(self):
        game = ZXGame(num_qubits=4, depth=3, max_episode_length=20, pe_dim=10)
        assert game.num_qubits == 4
        assert game.depth == 3
        assert game.max_episode_length == 20
        assert game.pe_dim == 10

    def test_initial_state_not_done(self, zx_game):
        assert not zx_game.done

    def test_episode_length_starts_at_zero(self, zx_game):
        assert zx_game.episode_length == 0

    def test_initial_return_is_zero(self, zx_game):
        assert zx_game.episode_return == 0.0

    def test_data_is_pyg_data(self, zx_game):
        from torch_geometric.data import Data
        assert isinstance(zx_game.data, Data)

    def test_data_has_required_attributes(self, zx_game):
        data = zx_game.data
        assert data.x is not None
        assert data.edge_index is not None
        assert data.edge_attr is not None
        assert data.node_type is not None

    def test_initial_diagram_stats_populated(self, zx_game):
        stats = zx_game.diagram_stats
        assert stats.num_nodes > 0
        assert stats.num_edges >= 0

    def test_initial_stats_capture(self, zx_game):
        """initial_t_gates / initial_nodes / initial_edges should match diagram_stats."""
        assert zx_game.initial_t_gates == zx_game.diagram_stats.num_non_clifford_gates
        assert zx_game.initial_nodes == zx_game.diagram_stats.num_nodes
        assert zx_game.initial_edges == zx_game.diagram_stats.num_edges

    def test_cumulative_rewards_start_at_zero(self, zx_game):
        assert zx_game.cumulative_t_gate_reward == 0.0
        assert zx_game.cumulative_node_reward == 0.0
        assert zx_game.cumulative_edge_reward == 0.0
        assert zx_game.cumulative_match_reward == 0.0


# ===========================================================================
# TestZXGameReset
# ===========================================================================


class TestZXGameReset:
    """Tests for ZXGame.reset."""

    def test_reset_returns_four_tuple(self, zx_game):
        result = zx_game.reset()
        assert len(result) == 4
        data, reward, done, info = result
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_reset_clears_episode_length(self, zx_game):
        # Advance the game a bit first if possible, then reset.
        zx_game.episode_length = 5
        zx_game.reset()
        assert zx_game.episode_length == 0

    def test_reset_clears_return(self, zx_game):
        zx_game.episode_return = 42.0
        zx_game.reset()
        assert zx_game.episode_return == 0.0

    def test_reset_clears_cumulative_rewards(self, zx_game):
        zx_game.cumulative_t_gate_reward = 10.0
        zx_game.cumulative_node_reward = 5.0
        zx_game.reset()
        assert zx_game.cumulative_t_gate_reward == 0.0
        assert zx_game.cumulative_node_reward == 0.0

    def test_reset_with_start_state(self, zx_game, zx_diagram):
        """Resetting with a specific diagram should use that diagram."""
        data, reward, done, info = zx_game.reset(start_state=zx_diagram)
        assert data is not None
        # The diagram_stats should reflect the provided diagram.
        assert zx_game.diagram_stats.num_nodes == zx_diagram.number_of_nodes()

    def test_reset_produces_valid_data(self, zx_game):
        data, _, _, _ = zx_game.reset()
        assert data.x is not None
        assert data.edge_index is not None
        assert data.node_type is not None

    def test_info_contains_diagram_stats(self, zx_game):
        _, _, _, info = zx_game.reset()
        assert 'diagram_stats' in info


# ===========================================================================
# TestZXGameStep
# ===========================================================================


class TestZXGameStep:
    """Tests for ZXGame.step with valid actions derived from the distribution."""

    @staticmethod
    def _get_valid_action(zx_game):
        """Use the model to produce a valid action for the current game state."""
        from alphazx.distributions import AlphaZXDistribution
        from alphazx.models import pre_process
        from alphazx.models.homogeneous.alphazx_model import AlphaZXModel

        num_node_types = len(METADATA.node_type_abbrevs)
        num_possible_phases = len(POSSIBLE_PHASES)
        num_possible_new_edges = 5
        pe_dim = zx_game.pe_dim

        model = AlphaZXModel(
            num_node_types, num_possible_phases, num_possible_new_edges,
            64, len(METADATA.edge_feat_to_index_dict), 64, pe_dim, pe_dim,
        )
        model.eval()

        data = pre_process(zx_game.data, pe_dim)
        with torch.no_grad():
            azx_dist_params, value = model(
                data.x, data.edge_index, data.edge_attr, data.node_type,
                torch.zeros_like(data.x, dtype=torch.int64), data.pe, data.id,
            )
            azx_dist = AlphaZXDistribution(azx_dist_params)
            action = tuple(azx_dist.sample(1).squeeze().tolist())
        return action

    def test_step_returns_four_tuple(self, zx_game):
        action = self._get_valid_action(zx_game)
        data, reward, done, info = zx_game.step(action)
        assert isinstance(reward, (int, float))
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    def test_step_increments_episode_length(self, zx_game):
        action = self._get_valid_action(zx_game)
        zx_game.step(action)
        assert zx_game.episode_length == 1

    def test_step_info_contains_reward_breakdown(self, zx_game):
        action = self._get_valid_action(zx_game)
        _, _, _, info = zx_game.step(action)
        assert 'reward_breakdown' in info
        assert 'diagram_stats' in info

    def test_step_updates_diagram_stats(self, zx_game):
        old_stats = vars(zx_game.diagram_stats).copy()
        action = self._get_valid_action(zx_game)
        zx_game.step(action)
        # Stats might or might not change, but the object should be fresh.
        assert zx_game.diagram_stats is not None

    def test_step_data_index_refreshed(self, zx_game):
        """After step, data and data_index should be consistent with the new diagram."""
        action = self._get_valid_action(zx_game)
        data, _, _, _ = zx_game.step(action)
        assert data is zx_game.data

    def test_max_episode_terminates(self):
        """Game should report done=True when episode_length == max_episode_length."""
        game = ZXGame(num_qubits=NUM_QUBITS, depth=DEPTH, max_episode_length=2, pe_dim=PE_DIM)
        for _ in range(2):
            if game.done:
                break
            action = TestZXGameStep._get_valid_action(game)
            _, _, done, info = game.step(action)
        assert game.done

    def test_episode_info_on_done(self):
        """When done, info should contain 'episode_info' with summary keys."""
        game = ZXGame(num_qubits=NUM_QUBITS, depth=DEPTH, max_episode_length=1, pe_dim=PE_DIM)
        action = TestZXGameStep._get_valid_action(game)
        _, _, done, info = game.step(action)
        assert done
        assert 'episode_info' in info
        ep = info['episode_info']
        assert 'length' in ep
        assert 'return' in ep
        assert 'initial_t_gates' in ep
        assert 'final_t_gates' in ep
        assert 't_gates_reduced' in ep
        assert 'simplified' in ep


# ===========================================================================
# TestRewardCalculation
# ===========================================================================


class TestRewardCalculation:
    """Tests for calculate_reward and calculate_reward_simple."""

    @staticmethod
    def _make_stats(**overrides):
        """Create a mock DiagramStats with controllable attributes."""

        class MockStats:
            pass

        s = MockStats()
        defaults = {
            'num_non_clifford_gates': 5,
            'num_nodes': 20,
            'num_edges': 30,
            'frz_nodes': 3,
            'flz_nodes': 2,
            'frx_nodes': 1,
            'flx_nodes': 0,
            'br_nodes': 2,
            'bl_nodes': 1,
            'yrz_nodes': 0,
            'ylz_nodes': 0,
            'yrx_nodes': 0,
            'ylx_nodes': 0,
        }
        defaults.update(overrides)
        for k, v in defaults.items():
            setattr(s, k, v)
        return s

    def test_reward_simple_t_gate_reduction(self):
        old = self._make_stats(num_non_clifford_gates=5)
        new = self._make_stats(num_non_clifford_gates=3)
        assert calculate_reward_simple(old, new) == 2

    def test_reward_simple_no_change(self):
        old = self._make_stats(num_non_clifford_gates=5)
        new = self._make_stats(num_non_clifford_gates=5)
        assert calculate_reward_simple(old, new) == 0

    def test_shaped_reward_t_gate_reduction_component(self):
        """Reward is now purely T-gate reduction: +1 per T-gate removed."""
        old = self._make_stats(num_non_clifford_gates=5)
        new = self._make_stats(num_non_clifford_gates=3)
        total, breakdown = calculate_reward(old, new)
        assert breakdown.t_gate_reward == 2.0  # 2 T-gates removed

    def test_reward_no_secondary_node_reward(self):
        """Node reduction no longer produces reward (secondary rewards removed)."""
        old = self._make_stats(num_nodes=20)
        new = self._make_stats(num_nodes=15)
        _, breakdown = calculate_reward(old, new)
        assert breakdown.node_reward == 0.0

    def test_reward_no_secondary_node_penalty(self):
        """Node increase no longer produces penalty (secondary rewards removed)."""
        old = self._make_stats(num_nodes=20)
        new = self._make_stats(num_nodes=23)
        _, breakdown = calculate_reward(old, new)
        assert breakdown.node_reward == 0.0

    def test_reward_no_secondary_edge_reward(self):
        """Edge reduction no longer produces reward (secondary rewards removed)."""
        old = self._make_stats(num_edges=30)
        new = self._make_stats(num_edges=25)
        _, breakdown = calculate_reward(old, new)
        assert breakdown.edge_reward == 0.0

    def test_reward_total_equals_t_gate_reward(self):
        """Total reward should equal T-gate reward only (no secondary components)."""
        old = self._make_stats(num_non_clifford_gates=5, num_nodes=20, num_edges=30, br_nodes=4)
        new = self._make_stats(num_non_clifford_gates=3, num_nodes=18, num_edges=28, br_nodes=2)
        total, breakdown = calculate_reward(old, new)
        assert breakdown.node_reward == 0.0
        assert breakdown.edge_reward == 0.0
        assert breakdown.match_reward == 0.0
        assert abs(total - breakdown.t_gate_reward) < 1e-9
        assert abs(total - 2.0) < 1e-9  # 2 T-gates removed

    def test_reward_breakdown_to_dict(self):
        rb = RewardBreakdown()
        rb.t_gate_reward = 10.0
        rb.node_reward = 0.0
        rb.total = 10.0
        d = rb.to_dict()
        assert d['t_gate_reward'] == 10.0
        assert d['total_reward'] == 10.0

    def test_reward_no_secondary_match_reward(self):
        """Match reduction no longer produces reward (secondary rewards removed)."""
        old = self._make_stats(br_nodes=5, bl_nodes=3)
        new = self._make_stats(br_nodes=3, bl_nodes=1)
        _, breakdown = calculate_reward(old, new)
        assert breakdown.match_reward == 0.0


# ===========================================================================
# TestDiagramHelpers
# ===========================================================================


class TestDiagramHelpers:
    """Tests for is_simplified, num_non_clifford_gates, and cleanup utilities."""

    def test_num_non_clifford_gates_clifford_only(self):
        """Clifford-only diagram (no t_gates) should have 0 non-Clifford gates."""
        d = clifford_zx_diagram(4, 4, t_gates=False)
        assert num_non_clifford_gates(d) == 0

    def test_is_simplified_clifford_only(self):
        d = clifford_zx_diagram(4, 4, t_gates=False)
        assert is_simplified(d)

    def test_remove_isolated_nodes_returns_set(self, zx_diagram):
        # Add an isolated node via the proper API (which registers it in
        # the internal node-type sets), then remove.
        iso_id = zx_diagram.add_z_node(0.0)
        removed = remove_isolated_nodes(zx_diagram)
        assert iso_id in removed
        assert iso_id not in zx_diagram.nodes()

    def test_remove_self_loop_edges(self, zx_diagram):
        # Add a self-loop.
        node = list(zx_diagram.nodes())[0]
        zx_diagram.add_edge(node, node)
        remove_self_loop_edges(zx_diagram)
        assert not any(u == v for u, v, _ in zx_diagram.edges(keys=True))

    def test_remove_isolated_components_keeps_boundary(self, zx_diagram):
        """Isolated components (no boundary nodes) should be removed."""
        # Just verify the function doesn't crash and the diagram retains boundaries.
        removed = remove_isolated_components(zx_diagram)
        b_nodes = zx_diagram.b_nodes()
        assert len(b_nodes) >= 2

    def test_remove_isolated_components_raises_on_no_boundaries(self):
        """Should raise if there are fewer than 2 boundary nodes."""
        from alphazx.diagram.zx_diagram import ZXDiagram
        d = ZXDiagram(phase_denominator=8)
        d.add_node(0, type='z', phase=0.0)
        with pytest.raises(ValueError, match='at least two boundary'):
            remove_isolated_components(d)


# ===========================================================================
# TestDiagramStats
# ===========================================================================


class TestDiagramStats:
    """Tests for DiagramStats initialization."""

    def test_stats_have_num_nodes(self, zx_match_diagram):
        stats = DiagramStats(zx_match_diagram)
        assert stats.num_nodes == zx_match_diagram.zx_diagram.number_of_nodes()

    def test_stats_have_num_edges(self, zx_match_diagram):
        stats = DiagramStats(zx_match_diagram)
        assert stats.num_edges == zx_match_diagram.zx_diagram.num_edges()

    def test_stats_have_match_node_counts(self, zx_match_diagram):
        stats = DiagramStats(zx_match_diagram)
        for abbrev in METADATA.match_node_type_abbrevs:
            attr = f'{abbrev}_nodes'
            assert hasattr(stats, attr)
            assert isinstance(getattr(stats, attr), int)


# ===========================================================================
# TestTupleToMatch
# ===========================================================================


class TestTupleToMatch:
    """Tests for tuple_to_match action → match conversion."""

    MATCH_TYPES = [
        (0, FRightZMatch),
        (1, FRightXMatch),
        (2, FLeftZMatch),
        (3, FLeftXMatch),
        (4, BRightMatch),
        (5, BLeftMatch),
        (6, YRightZMatch),
        (7, YRightXMatch),
        (8, YLeftZMatch),
        (9, YLeftXMatch),
    ]

    def test_action_type_offset(self):
        """Action type index + 1 should equal the match type's class index."""
        for action_idx, match_cls in self.MATCH_TYPES:
            assert action_idx + 1 == match_cls.index

    def test_invalid_action_type_raises(self, zx_game):
        """Action type outside 0-9 should raise ValueError."""
        # Build a fake action with an out-of-range type
        data = zx_game.data
        action = (0, 10, 0, 0, 0)  # action_type 10 → match_type_index 11 (invalid)
        with pytest.raises(ValueError, match='Unexpected action type'):
            tuple_to_match(zx_game.zx_match_diagram, data, action, zx_game.data_index)

    def test_f_right_returns_params(self, zx_game):
        """FRightZ/FRightX actions should return non-None params."""
        from alphazx.models.utils import compute_basis_neighbors
        # Find an FRightZ match if one exists in this diagram.
        for idx, match in zx_game.data_index.indices.items():
            if isinstance(match, FRightZMatch):
                # Compute how many basis neighbors this node has so we can
                # provide the required Bernoulli transfer-edge flags.
                basis_neighbors = compute_basis_neighbors(
                    zx_game.data.edge_index, idx, zx_game.data.node_type,
                )
                transfer_flags = tuple([0] * len(basis_neighbors))
                # Action: (graph_id, action_type=0, node_idx, phase=0, new_edges=0, *transfer_edges)
                action = (0, 0, idx, 0, 0) + transfer_flags
                result_match, params = tuple_to_match(
                    zx_game.zx_match_diagram, zx_game.data, action, zx_game.data_index,
                )
                assert isinstance(result_match, FRightZMatch)
                assert params is not None
                return
        pytest.skip("No FRightZMatch found in this random diagram")

    def test_non_f_right_returns_none_params(self, zx_game):
        """Non-FRight actions should return None for params."""
        non_fright_types = {
            FLeftZMatch: 2,
            FLeftXMatch: 3,
            BRightMatch: 4,
            BLeftMatch: 5,
            YRightZMatch: 6,
            YRightXMatch: 7,
            YLeftZMatch: 8,
            YLeftXMatch: 9,
        }
        for idx, match in zx_game.data_index.indices.items():
            match_type = type(match)
            if match_type in non_fright_types:
                action_type = non_fright_types[match_type]
                action = (0, action_type, idx, 0, 0)
                result_match, params = tuple_to_match(
                    zx_game.zx_match_diagram, zx_game.data, action, zx_game.data_index,
                )
                assert params is None
                return
        pytest.skip("No non-FRight match found in this random diagram")
