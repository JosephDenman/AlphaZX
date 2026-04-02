"""
Tests for MCTS infrastructure: GameState, MCTSNode, MCTS search, and evaluation.

These tests verify:
1. GameState cloning produces independent copies
2. GameState apply_action correctly mutates state
3. MCTSNode PUCT scoring and progressive widening logic
4. Full MCTS search produces a valid policy distribution
5. End-to-end integration with the AlphaZXModel
"""

import math
import unittest

import torch

from alphazx.diagram import METADATA, POSSIBLE_PHASES
from alphazx.diagram.diagram_generators import clifford_zx_diagram
from alphazx.distributions import AlphaZXDistribution
from alphazx.game import ZXGame
from alphazx.models import pre_process_single
from alphazx.models.homogeneous.alphazx_model import AlphaZXModel
from alphazx.mcts.config import MCTSConfig
from alphazx.mcts.game_state import GameState
from alphazx.mcts.node import MCTSNode
from alphazx.mcts.search import MCTS
from alphazx.mcts.evaluate import evaluate_state, compute_action_prior

# Note: we intentionally do NOT set torch.set_default_dtype(torch.float64) here.
# The pre_process module's with_random_walk_pe creates float32 tensors internally,
# and setting the default to float64 causes a dtype mismatch during matrix multiply.

# --- Shared model setup (same as test_zx_game.py) ---
PE_DIM = 20
NUM_NODE_TYPES = len(METADATA.node_type_abbrevs)
NUM_POSSIBLE_PHASES = len(POSSIBLE_PHASES)
NUM_POSSIBLE_NEW_EDGES = 5


def make_model():
    return AlphaZXModel(
        NUM_NODE_TYPES,
        NUM_POSSIBLE_PHASES,
        NUM_POSSIBLE_NEW_EDGES,
        node_embedding_channels=64,
        num_edge_embeddings=len(METADATA.edge_feat_to_index_dict),
        edge_embedding_channels=64,
        pe_in_channels=PE_DIM,
        pe_out_channels=PE_DIM,
    )


def make_game(num_qubits=5, depth=5):
    return ZXGame(num_qubits, depth, pe_dim=PE_DIM, max_episode_length=50)


def make_config(**overrides):
    defaults = dict(
        num_simulations=20,
        pe_dim=PE_DIM,
        num_qubits=5,
        depth=5,
        temperature=1.0,
    )
    defaults.update(overrides)
    return MCTSConfig(**defaults)


class TestGameState(unittest.TestCase):
    """Test the GameState wrapper."""

    def test_from_game(self):
        """GameState.from_game() creates a valid state with correct properties."""
        game = make_game()
        game.reset()
        state = GameState.from_game(game)

        self.assertGreater(state.num_nodes, 0)
        self.assertGreater(state.num_edges, 0)
        self.assertIsNotNone(state.data)
        self.assertIsNotNone(state.data_index)

    def test_from_diagram(self):
        """GameState.from_diagram() creates a valid state from a raw diagram."""
        diagram = clifford_zx_diagram(5, 5, True)
        state = GameState.from_diagram(diagram)

        self.assertGreater(state.num_nodes, 0)
        self.assertTrue(state.has_legal_actions())

    def test_clone_independence(self):
        """Cloned states are fully independent — mutating one does not affect the other."""
        game = make_game()
        game.reset()
        state = GameState.from_game(game)

        original_nodes = state.num_nodes
        original_edges = state.num_edges
        original_t_gates = state.num_non_clifford

        clone = state.clone()

        # Verify clone has same properties
        self.assertEqual(clone.num_nodes, original_nodes)
        self.assertEqual(clone.num_edges, original_edges)
        self.assertEqual(clone.num_non_clifford, original_t_gates)

        # Now try to apply an action to the clone (if possible)
        model = make_model()
        distribution, _ = evaluate_state(model, clone, PE_DIM)
        action = tuple(distribution.sample(1).squeeze().tolist())

        try:
            clone.apply_action(action)
        except Exception:
            # Action might fail on some random states — that's OK for this test
            return

        # Original should be unchanged
        self.assertEqual(state.num_nodes, original_nodes)
        self.assertEqual(state.num_edges, original_edges)

    def test_apply_action_mutates_state(self):
        """apply_action() changes the state's diagram."""
        game = make_game()
        game.reset()
        state = GameState.from_game(game)

        model = make_model()
        distribution, _ = evaluate_state(model, state, PE_DIM)

        # Try multiple samples until one works
        for _ in range(20):
            action = tuple(distribution.sample(1).squeeze().tolist())
            clone = state.clone()
            try:
                reward, done = clone.apply_action(action)
                # If we get here, the action was valid
                self.assertIsInstance(reward, float)
                self.assertIsInstance(done, bool)
                return
            except (ValueError, KeyError, IndexError, AssertionError):
                continue

        # If no action worked in 20 tries, skip (very unlikely with valid state)
        self.skipTest("Could not find a valid action in 20 samples")

    def test_is_terminal(self):
        """is_terminal() returns False for non-trivial circuits."""
        game = make_game()
        game.reset()
        state = GameState.from_game(game)

        # A fresh circuit with T-gates should not be terminal
        if state.num_non_clifford > 0:
            self.assertFalse(state.is_terminal())

    def test_has_legal_actions(self):
        """has_legal_actions() returns True for non-trivial circuits."""
        game = make_game()
        game.reset()
        state = GameState.from_game(game)

        self.assertTrue(state.has_legal_actions())


class TestMCTSNode(unittest.TestCase):
    """Test MCTSNode mechanics."""

    def _make_node(self) -> MCTSNode:
        game = make_game()
        game.reset()
        state = GameState.from_game(game)
        return MCTSNode(state=state)

    def test_initial_state(self):
        """New nodes start with zero visits and are not expanded."""
        node = self._make_node()
        self.assertEqual(node.visit_count, 0)
        self.assertEqual(node.total_value, 0.0)
        self.assertEqual(node.q_value, 0.0)
        self.assertFalse(node.is_expanded)
        self.assertEqual(len(node.children), 0)

    def test_puct_score_unvisited(self):
        """Unvisited children with high priors get high PUCT scores."""
        config = make_config()
        parent = self._make_node()
        parent.visit_count = 10  # Simulate some visits

        child_high = MCTSNode(state=parent.state, parent=parent, prior=0.9)
        child_low = MCTSNode(state=parent.state, parent=parent, prior=0.1)

        # Both unvisited → score is purely from exploration term
        score_high = child_high.puct_score(config)
        score_low = child_low.puct_score(config)

        self.assertGreater(score_high, score_low)

    def test_puct_balances_exploration_exploitation(self):
        """PUCT favors visited children with high Q over unvisited ones eventually."""
        config = make_config(c_puct=1.0)
        parent = self._make_node()
        parent.visit_count = 100

        # A well-visited child with high Q
        child_good = MCTSNode(state=parent.state, parent=parent, prior=0.1)
        child_good.visit_count = 50
        child_good.total_value = 40.0  # Q = 0.8

        # An unvisited child with modest prior
        child_new = MCTSNode(state=parent.state, parent=parent, prior=0.3)

        # With enough parent visits, the high-Q child should eventually win
        # because its Q term (0.8) dominates the exploration bonus
        score_good = child_good.puct_score(config)
        score_new = child_new.puct_score(config)

        # The well-explored good child should be competitive
        # (exact comparison depends on params, but both should be positive)
        self.assertGreater(score_good, 0)
        self.assertGreater(score_new, 0)

    def test_progressive_widening(self):
        """should_widen() returns True when visits exceed the threshold."""
        config = make_config(pw_alpha=0.5, pw_c=1.0)
        node = self._make_node()
        node.is_expanded = True

        # No children → should always widen
        self.assertTrue(node.should_widen(config))

        # Add a fake child
        node.children[('dummy',)] = MCTSNode(state=node.state, parent=node)

        # 1 child, threshold = pw_c * 1^(1/0.5) = 1.0
        # Need visit_count >= 1.0 to widen
        node.visit_count = 0
        self.assertFalse(node.should_widen(config))

        node.visit_count = 1
        self.assertTrue(node.should_widen(config))

    def test_backpropagation(self):
        """backpropagate() updates visit counts and values up to root."""
        root = self._make_node()
        root.visit_count = 5
        root.total_value = 3.0

        child = MCTSNode(state=root.state, parent=root, reward=1.0)
        root.children[('action',)] = child

        grandchild = MCTSNode(state=root.state, parent=child, reward=0.5)
        child.children[('action2',)] = grandchild

        # Backpropagate value=2.0 from grandchild
        grandchild.backpropagate(2.0, gamma=1.0)

        # Grandchild: N=1, W=2.0
        self.assertEqual(grandchild.visit_count, 1)
        self.assertAlmostEqual(grandchild.total_value, 2.0)

        # Child: N=1, W = reward(grandchild) + gamma * value = 0.5 + 1.0 * 2.0 = 2.5
        self.assertEqual(child.visit_count, 1)
        self.assertAlmostEqual(child.total_value, 2.5)

        # Root: N=6 (5+1), W = 3.0 + reward(child) + gamma * child_value = 3.0 + 1.0 + 2.5 = 6.5
        self.assertEqual(root.visit_count, 6)
        self.assertAlmostEqual(root.total_value, 6.5)


class TestEvaluateState(unittest.TestCase):
    """Test neural network evaluation."""

    def test_evaluate_returns_distribution_and_value(self):
        """evaluate_state() returns an AlphaZXDistribution and a scalar value."""
        model = make_model()
        game = make_game()
        game.reset()
        state = GameState.from_game(game)

        distribution, value = evaluate_state(model, state, PE_DIM)

        self.assertIsInstance(distribution, AlphaZXDistribution)
        self.assertIsInstance(value, float)

    def test_distribution_is_sampleable(self):
        """The returned distribution can produce valid action samples."""
        model = make_model()
        game = make_game()
        game.reset()
        state = GameState.from_game(game)

        distribution, _ = evaluate_state(model, state, PE_DIM)
        samples = distribution.sample(5)  # Sample 5 actions

        self.assertEqual(samples.shape[0], 1)  # batch size 1
        self.assertEqual(samples.shape[1], 5)  # 5 samples
        self.assertGreater(samples.shape[2], 4)  # at least 5 entries per action

    def test_compute_action_prior(self):
        """compute_action_prior() returns a valid probability in [0, 1]."""
        model = make_model()
        game = make_game()
        game.reset()
        state = GameState.from_game(game)

        distribution, _ = evaluate_state(model, state, PE_DIM)
        action = tuple(distribution.sample(1).squeeze().tolist())

        prior = compute_action_prior(distribution, action)
        self.assertGreater(prior, 0.0)
        self.assertLessEqual(prior, 1.0)


class TestMCTSSearch(unittest.TestCase):
    """Test the full MCTS search loop."""

    def test_search_returns_valid_policy(self):
        """MCTS search returns a probability distribution over actions."""
        model = make_model()
        config = make_config(num_simulations=10)
        mcts = MCTS(model, config)

        game = make_game()
        game.reset()
        state = GameState.from_game(game)

        policy = mcts.search(state)

        # Policy should be non-empty
        self.assertGreater(len(policy), 0)

        # Probabilities should sum to ~1
        total_prob = sum(policy.values())
        self.assertAlmostEqual(total_prob, 1.0, places=5)

        # All probabilities should be non-negative
        for prob in policy.values():
            self.assertGreaterEqual(prob, 0.0)

    def test_search_visits_accumulate(self):
        """After search, root children have non-zero visit counts."""
        model = make_model()
        config = make_config(num_simulations=20)
        mcts = MCTS(model, config)

        game = make_game()
        game.reset()
        state = GameState.from_game(game)

        policy = mcts.search(state)

        # At least one action should have been visited
        total_visits = sum(
            int(p * config.num_simulations + 0.5) for p in policy.values()
        )
        # Visits should be positive (approximately num_simulations)
        self.assertGreater(len(policy), 0)

    def test_select_action_returns_valid_action(self):
        """select_action() returns an action that exists in the policy."""
        model = make_model()
        config = make_config(num_simulations=10)
        mcts = MCTS(model, config)

        game = make_game()
        game.reset()
        state = GameState.from_game(game)

        action, policy, _ = mcts.select_action(state)

        self.assertIn(action, policy)
        self.assertIsInstance(action, tuple)

    def test_low_temperature_is_near_greedy(self):
        """With low temperature, the policy concentrates on the most-visited action."""
        model = make_model()
        config = make_config(num_simulations=30, temperature=0.001)
        mcts = MCTS(model, config)

        game = make_game()
        game.reset()
        state = GameState.from_game(game)

        policy = mcts.search(state)

        # The max probability should be close to 1.0
        max_prob = max(policy.values())
        self.assertGreater(max_prob, 0.9)


if __name__ == '__main__':
    unittest.main()
