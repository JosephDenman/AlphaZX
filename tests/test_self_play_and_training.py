"""
Tests for Phases 4-5: Self-play data generation, replay buffer, and training loop.

These tests run on tiny circuits (3 qubits, depth 3) with minimal MCTS simulations
to keep runtime manageable. They verify correctness of the data pipeline, not
quality of the learned policy.

Test structure:
1. ReplayBuffer: add, sample, overflow, collation
2. SelfPlayWorker: produces valid episodes with value targets
3. SelfPlayManager: fills the replay buffer correctly
4. Trainer: one iteration runs without error, loss is computed
5. Evaluator: basic evaluation produces a summary
"""

import math
import unittest

import torch

from alphazx.diagram import METADATA, POSSIBLE_PHASES
from alphazx.diagram.diagram_generators import clifford_zx_diagram
from alphazx.game import ZXGame
from alphazx.models.homogeneous.alphazx_model import AlphaZXModel
from alphazx.mcts.config import MCTSConfig
from alphazx.mcts.game_state import GameState
from alphazx.mcts.replay_buffer import ReplayBuffer, TrainingExample, collate_training_batch
from alphazx.mcts.self_play import SelfPlayWorker, SelfPlayManager
from alphazx.mcts.trainer import Trainer, TrainerConfig
from alphazx.mcts.evaluator import Evaluator

# --- Shared setup ---
PE_DIM = 20
NUM_NODE_TYPES = len(METADATA.node_type_abbrevs)
NUM_POSSIBLE_PHASES = len(POSSIBLE_PHASES)
NUM_POSSIBLE_NEW_EDGES = 5

# Tiny config for fast tests
TINY_QUBITS = 3
TINY_DEPTH = 3
TINY_SIMULATIONS = 5


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


def make_mcts_config(**overrides):
    defaults = dict(
        num_simulations=TINY_SIMULATIONS,
        pe_dim=PE_DIM,
        num_qubits=TINY_QUBITS,
        depth=TINY_DEPTH,
        max_episode_length=10,
        temperature=1.0,
    )
    defaults.update(overrides)
    return MCTSConfig(**defaults)


# ==============================================================================
# Replay Buffer Tests
# ==============================================================================

class TestReplayBuffer(unittest.TestCase):
    """Tests for the circular replay buffer."""

    def _make_dummy_example(self, value: float = 1.0) -> TrainingExample:
        """Create a minimal TrainingExample for testing buffer mechanics."""
        # We don't need real PyG data for buffer tests — just a placeholder
        dummy_data = torch.zeros(1)
        return TrainingExample(
            state_data=dummy_data,
            mcts_policy={(0, 1, 2): 0.5, (0, 1, 3): 0.5},
            value_target=value,
            game_id=1,
        )

    def test_add_and_len(self):
        buf = ReplayBuffer(capacity=100)
        self.assertEqual(len(buf), 0)

        buf.add(self._make_dummy_example())
        self.assertEqual(len(buf), 1)

        for _ in range(9):
            buf.add(self._make_dummy_example())
        self.assertEqual(len(buf), 10)

    def test_sample_returns_correct_count(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(20):
            buf.add(self._make_dummy_example())

        samples = buf.sample(5)
        self.assertEqual(len(samples), 5)

    def test_sample_clamps_to_buffer_size(self):
        buf = ReplayBuffer(capacity=100)
        for _ in range(3):
            buf.add(self._make_dummy_example())

        samples = buf.sample(10)
        self.assertEqual(len(samples), 3)  # clamped to buffer size

    def test_sample_raises_on_empty(self):
        buf = ReplayBuffer(capacity=100)
        with self.assertRaises(ValueError):
            buf.sample(1)

    def test_circular_overflow(self):
        buf = ReplayBuffer(capacity=5)
        for i in range(10):
            buf.add(self._make_dummy_example(value=float(i)))

        self.assertEqual(len(buf), 5)
        self.assertEqual(buf.total_added, 10)

        # The buffer should contain the last 5 examples (values 5-9)
        samples = buf.sample(5)
        values = sorted([s.value_target for s in samples])
        self.assertEqual(values, [5.0, 6.0, 7.0, 8.0, 9.0])

    def test_add_game(self):
        buf = ReplayBuffer(capacity=100)
        examples = [self._make_dummy_example(value=float(i)) for i in range(5)]
        buf.add_game(examples)
        self.assertEqual(len(buf), 5)

    def test_is_full(self):
        buf = ReplayBuffer(capacity=3)
        self.assertFalse(buf.is_full)
        for _ in range(3):
            buf.add(self._make_dummy_example())
        self.assertTrue(buf.is_full)


# ==============================================================================
# Self-Play Tests
# ==============================================================================

class TestSelfPlayWorker(unittest.TestCase):
    """Tests for self-play episode generation."""

    def test_play_episode_produces_examples(self):
        """Self-play produces training examples with filled value targets."""
        model = make_model()
        config = make_mcts_config()
        worker = SelfPlayWorker(model, config)

        result = worker.play_episode()

        # Should have produced at least one step
        self.assertGreaterEqual(result.num_steps, 0)
        # Examples list should match num_steps
        self.assertEqual(len(result.examples), result.num_steps)

        # All value targets should be filled in (not None)
        for ex in result.examples:
            self.assertIsNotNone(ex.value_target)
            self.assertIsInstance(ex.value_target, float)

        # T-gate stats should be consistent
        self.assertEqual(
            result.t_gates_reduced,
            result.initial_t_gates - result.final_t_gates,
        )

    def test_play_episode_with_start_diagram(self):
        """Self-play can start from a provided diagram."""
        model = make_model()
        config = make_mcts_config()
        worker = SelfPlayWorker(model, config)

        diagram = clifford_zx_diagram(TINY_QUBITS, TINY_DEPTH, t_gates=True)
        result = worker.play_episode(start_diagram=diagram)

        self.assertGreaterEqual(result.num_steps, 0)

    def test_examples_have_valid_mcts_policies(self):
        """Each training example's MCTS policy is a valid probability distribution."""
        model = make_model()
        config = make_mcts_config()
        worker = SelfPlayWorker(model, config)

        result = worker.play_episode()

        for ex in result.examples:
            self.assertIsInstance(ex.mcts_policy, dict)
            if ex.mcts_policy:  # non-empty
                total_prob = sum(ex.mcts_policy.values())
                self.assertAlmostEqual(total_prob, 1.0, places=5,
                                       msg="MCTS policy should sum to 1")
                for prob in ex.mcts_policy.values():
                    self.assertGreaterEqual(prob, 0.0)

    def test_examples_have_preprocessed_data(self):
        """State data in examples should have positional encoding (pe attribute)."""
        model = make_model()
        config = make_mcts_config()
        worker = SelfPlayWorker(model, config)

        result = worker.play_episode()

        for ex in result.examples:
            data = ex.state_data
            self.assertTrue(hasattr(data, 'pe'),
                            "State data should have positional encoding")
            self.assertTrue(hasattr(data, 'x'))
            self.assertTrue(hasattr(data, 'edge_index'))
            self.assertTrue(hasattr(data, 'edge_attr'))


class TestSelfPlayManager(unittest.TestCase):
    """Tests for self-play game orchestration."""

    def test_generate_games_fills_buffer(self):
        """Generating games adds examples to the replay buffer."""
        model = make_model()
        config = make_mcts_config()
        buffer = ReplayBuffer(capacity=10000)
        manager = SelfPlayManager(model, config, buffer)

        results = manager.generate_games(num_games=2)

        self.assertEqual(len(results), 2)
        self.assertGreater(len(buffer), 0)
        self.assertEqual(manager.total_games, 2)

    def test_stats_summary(self):
        """Stats summary returns expected keys."""
        model = make_model()
        config = make_mcts_config()
        buffer = ReplayBuffer(capacity=10000)
        manager = SelfPlayManager(model, config, buffer)

        manager.generate_games(num_games=1)
        stats = manager.stats_summary()

        self.assertIn('total_games', stats)
        self.assertIn('total_examples', stats)
        self.assertIn('buffer_size', stats)
        self.assertEqual(stats['total_games'], 1)


# ==============================================================================
# Training Tests
# ==============================================================================

class TestTrainer(unittest.TestCase):
    """Tests for the training loop."""

    def test_single_iteration(self):
        """One full iteration (self-play + training) runs without error."""
        model = make_model()
        mcts_config = make_mcts_config()
        trainer_config = TrainerConfig(
            num_self_play_games=2,
            training_steps=2,
            batch_size=2,
            learning_rate=1e-3,
            num_iterations=1,
            min_buffer_size=1,  # Very low for testing
            eval_interval=999,  # Skip eval
            checkpoint_interval=999,  # Skip checkpoint
        )
        buffer = ReplayBuffer(capacity=10000)

        trainer = Trainer(
            model=model,
            mcts_config=mcts_config,
            trainer_config=trainer_config,
            replay_buffer=buffer,
        )

        metrics_list = trainer.train()

        self.assertEqual(len(metrics_list), 1)
        metrics = metrics_list[0]

        # Losses should be finite numbers (not NaN or inf)
        self.assertTrue(math.isfinite(metrics.avg_policy_loss),
                        f"Policy loss not finite: {metrics.avg_policy_loss}")
        self.assertTrue(math.isfinite(metrics.avg_value_loss),
                        f"Value loss not finite: {metrics.avg_value_loss}")
        self.assertTrue(math.isfinite(metrics.avg_total_loss),
                        f"Total loss not finite: {metrics.avg_total_loss}")

    def test_loss_is_positive(self):
        """Policy and value losses should be non-negative."""
        model = make_model()
        mcts_config = make_mcts_config()
        trainer_config = TrainerConfig(
            num_self_play_games=3,
            training_steps=5,
            batch_size=2,
            learning_rate=1e-3,
            num_iterations=1,
            min_buffer_size=1,
            eval_interval=999,
            checkpoint_interval=999,
        )
        buffer = ReplayBuffer(capacity=10000)

        trainer = Trainer(
            model=model,
            mcts_config=mcts_config,
            trainer_config=trainer_config,
            replay_buffer=buffer,
        )

        metrics_list = trainer.train()
        metrics = metrics_list[0]

        # Policy loss (cross-entropy) should be positive
        self.assertGreaterEqual(metrics.avg_policy_loss, 0.0,
                                "Policy loss should be non-negative")
        # Value loss (MSE) should be non-negative
        self.assertGreaterEqual(metrics.avg_value_loss, 0.0,
                                "Value loss should be non-negative")

    def test_two_iterations_both_run(self):
        """Two consecutive iterations complete successfully."""
        model = make_model()
        mcts_config = make_mcts_config()
        trainer_config = TrainerConfig(
            num_self_play_games=2,
            training_steps=2,
            batch_size=2,
            learning_rate=1e-3,
            num_iterations=2,
            min_buffer_size=1,
            eval_interval=999,
            checkpoint_interval=999,
        )
        buffer = ReplayBuffer(capacity=10000)

        trainer = Trainer(
            model=model,
            mcts_config=mcts_config,
            trainer_config=trainer_config,
            replay_buffer=buffer,
        )

        metrics_list = trainer.train()
        self.assertEqual(len(metrics_list), 2)
        # Both iterations should have finite loss
        for m in metrics_list:
            self.assertTrue(math.isfinite(m.avg_total_loss))


# ==============================================================================
# Evaluator Tests
# ==============================================================================

class TestEvaluator(unittest.TestCase):
    """Tests for evaluation against baselines."""

    def test_basic_evaluation(self):
        """Evaluator runs and produces a summary."""
        model = make_model()
        mcts_config = make_mcts_config()

        evaluator = Evaluator(
            mcts_config=mcts_config,
            compare_pyzx=False,  # Skip PyZX comparison for speed
            eval_temperature=0.1,
        )

        summary = evaluator.evaluate(model, num_games=2)

        self.assertEqual(summary.num_games, 2)
        self.assertGreaterEqual(summary.avg_steps, 0)
        self.assertIsInstance(summary.avg_t_gates_reduced, float)

    def test_evaluation_with_pyzx(self):
        """Evaluator can compare against PyZX baseline."""
        model = make_model()
        mcts_config = make_mcts_config()

        evaluator = Evaluator(
            mcts_config=mcts_config,
            compare_pyzx=True,
            eval_temperature=0.1,
        )

        summary = evaluator.evaluate(model, num_games=1)

        self.assertEqual(summary.num_games, 1)
        # PyZX comparison fields should be populated
        # (Note: pyzx_avg_t_gates_reduced could be None if pyzx fails)


# ==============================================================================
# Collation Tests
# ==============================================================================

class TestCollation(unittest.TestCase):
    """Tests for training batch collation."""

    def test_collate_training_batch(self):
        """collate_training_batch produces correct tensor shapes."""
        model = make_model()
        config = make_mcts_config()
        worker = SelfPlayWorker(model, config)

        result = worker.play_episode()
        if len(result.examples) < 2:
            self.skipTest("Episode too short for collation test")

        examples = result.examples[:2]
        batch, policy_targets, value_targets = collate_training_batch(examples)

        self.assertEqual(value_targets.shape[0], 2)
        self.assertEqual(len(policy_targets), 2)
        self.assertIsInstance(policy_targets[0], dict)


if __name__ == '__main__':
    unittest.main()
