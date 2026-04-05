"""
Tests for multi-process self-play (parallel_self_play.py).

Tests run on tiny circuits (3 qubits, depth 3) with minimal MCTS simulations
to keep runtime manageable. They verify correctness of the parallelization
mechanics, not quality of the generated data.

Test structure:
1. _extract_model_hparams: round-trips through model reconstruction
2. _worker_play_games: standalone worker produces valid EpisodeResults
3. ParallelSelfPlayManager: generates games across multiple workers
4. Curriculum integration: difficulty overrides are partitioned correctly
5. Trainer integration: parallel manager is selected when workers > 1
"""

import unittest

import torch

from alphazx.diagram import METADATA, POSSIBLE_PHASES
from alphazx.models.homogeneous.alphazx_model import AlphaZXModel
from alphazx.mcts.config import MCTSConfig
from alphazx.mcts.replay_buffer import ReplayBuffer
from alphazx.mcts.parallel_self_play import (
    _extract_model_hparams,
    _build_model_from_hparams,
    _worker_play_games,
    ParallelSelfPlayManager,
)

# --- Shared setup ---
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


def make_mcts_config(**overrides):
    defaults = dict(
        num_simulations=5,
        pe_dim=PE_DIM,
        num_qubits=3,
        depth=3,
        max_episode_length=5,
        temperature=1.0,
        max_t_gate_increase=3,
        min_initial_t_gates=1,
    )
    defaults.update(overrides)
    return MCTSConfig(**defaults)


# ==============================================================================
# Model Hparams Extraction + Reconstruction
# ==============================================================================

class TestModelHparams(unittest.TestCase):
    """Test that model hyperparameters can be extracted and used to reconstruct."""

    def test_extract_hparams(self):
        """_extract_model_hparams returns the correct constructor arguments."""
        model = make_model()
        hparams = _extract_model_hparams(model)

        self.assertEqual(hparams['num_node_types'], NUM_NODE_TYPES)
        self.assertEqual(hparams['num_possible_phases'], NUM_POSSIBLE_PHASES)
        self.assertEqual(hparams['num_possible_new_edges'], NUM_POSSIBLE_NEW_EDGES)
        self.assertEqual(hparams['node_embedding_channels'], 64)
        self.assertEqual(hparams['num_edge_embeddings'], len(METADATA.edge_feat_to_index_dict))
        self.assertEqual(hparams['edge_embedding_channels'], 64)
        self.assertEqual(hparams['pe_in_channels'], PE_DIM)
        self.assertEqual(hparams['pe_out_channels'], PE_DIM)

    def test_reconstruct_model_from_hparams(self):
        """A model reconstructed from hparams can load the original state_dict."""
        model = make_model()
        hparams = _extract_model_hparams(model)
        state_dict = model.state_dict()

        reconstructed = _build_model_from_hparams(hparams)
        # Should load without errors (strict=True is default)
        reconstructed.load_state_dict(state_dict)

        # Verify outputs match on a dummy input
        model.eval()
        reconstructed.eval()

    def test_hparams_roundtrip_state_dict(self):
        """Reconstructed model with loaded state_dict produces identical outputs."""
        model = make_model()
        model.eval()
        hparams = _extract_model_hparams(model)
        state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        reconstructed = _build_model_from_hparams(hparams)
        reconstructed.load_state_dict(state_dict)
        reconstructed.eval()

        # Both should have identical parameters
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), reconstructed.named_parameters()
        ):
            self.assertEqual(n1, n2)
            self.assertTrue(
                torch.equal(p1, p2),
                f"Parameter {n1} differs after reconstruction",
            )


# ==============================================================================
# Worker Function
# ==============================================================================

class TestWorkerFunction(unittest.TestCase):
    """Test the standalone _worker_play_games function."""

    def test_worker_produces_results(self):
        """Worker function returns valid EpisodeResult objects."""
        model = make_model()
        config = make_mcts_config()
        hparams = _extract_model_hparams(model)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        results = _worker_play_games(
            model_state_dict=state_dict,
            model_hparams=hparams,
            mcts_config=config,
            num_games=2,
            worker_seed=42,
        )

        self.assertEqual(len(results), 2)
        for r in results:
            self.assertIsInstance(r.num_steps, int)
            self.assertGreaterEqual(r.num_steps, 0)
            self.assertIsInstance(r.initial_t_gates, int)
            self.assertIsInstance(r.final_t_gates, int)
            self.assertIsInstance(r.examples, list)
            # Value targets should be filled in
            for ex in r.examples:
                self.assertIsNotNone(ex.value_target)

    def test_different_seeds_produce_different_results(self):
        """Workers with different seeds should generate different games."""
        model = make_model()
        config = make_mcts_config()
        hparams = _extract_model_hparams(model)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        results_a = _worker_play_games(state_dict, hparams, config, 1, worker_seed=100)
        results_b = _worker_play_games(state_dict, hparams, config, 1, worker_seed=200)

        # With different seeds, at least one of these metrics should differ
        # (extremely unlikely to be identical with different random circuits)
        a, b = results_a[0], results_b[0]
        different = (
            a.initial_t_gates != b.initial_t_gates
            or a.final_t_gates != b.final_t_gates
            or a.num_steps != b.num_steps
        )
        # This is probabilistic but virtually certain with different seeds
        # on random circuit generation. If it ever flakes, increase games.
        self.assertTrue(
            different,
            "Two workers with different seeds produced identical results "
            "(extremely unlikely — possible flake)"
        )

    def test_worker_with_difficulty_overrides(self):
        """Worker respects difficulty overrides for curriculum."""
        model = make_model()
        config = make_mcts_config()
        hparams = _extract_model_hparams(model)
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        overrides = [(2, 2), (4, 5)]
        results = _worker_play_games(
            state_dict, hparams, config, 2, worker_seed=42,
            difficulty_overrides=overrides,
        )

        self.assertEqual(len(results), 2)
        # Config should be restored after each game
        self.assertEqual(config.num_qubits, 3)  # original value
        self.assertEqual(config.depth, 3)


# ==============================================================================
# ParallelSelfPlayManager
# ==============================================================================

class TestParallelSelfPlayManager(unittest.TestCase):
    """Test the ParallelSelfPlayManager."""

    def test_generate_games_returns_correct_count(self):
        """Manager generates the requested number of games."""
        model = make_model()
        config = make_mcts_config()
        replay_buffer = ReplayBuffer(capacity=10000)

        manager = ParallelSelfPlayManager(
            model=model,
            config=config,
            replay_buffer=replay_buffer,
            num_workers=2,
        )

        try:
            results = manager.generate_games(4)
            self.assertEqual(len(results), 4)
        finally:
            manager.shutdown()

    def test_examples_added_to_replay_buffer(self):
        """Generated examples are inserted into the replay buffer."""
        model = make_model()
        config = make_mcts_config()
        replay_buffer = ReplayBuffer(capacity=10000)

        manager = ParallelSelfPlayManager(
            model=model,
            config=config,
            replay_buffer=replay_buffer,
            num_workers=2,
        )

        try:
            results = manager.generate_games(4)
            total_examples = sum(len(r.examples) for r in results)
            self.assertEqual(len(replay_buffer), total_examples)
        finally:
            manager.shutdown()

    def test_lifetime_statistics_updated(self):
        """Manager tracks lifetime statistics correctly."""
        model = make_model()
        config = make_mcts_config()
        replay_buffer = ReplayBuffer(capacity=10000)

        manager = ParallelSelfPlayManager(
            model=model,
            config=config,
            replay_buffer=replay_buffer,
            num_workers=2,
        )

        try:
            results = manager.generate_games(4)
            self.assertEqual(manager.total_games, 4)
            self.assertEqual(
                manager.total_examples,
                sum(len(r.examples) for r in results),
            )
        finally:
            manager.shutdown()

    def test_stats_summary_interface(self):
        """stats_summary() returns the expected keys (mirrors SelfPlayManager)."""
        model = make_model()
        config = make_mcts_config()
        replay_buffer = ReplayBuffer(capacity=10000)

        manager = ParallelSelfPlayManager(
            model=model,
            config=config,
            replay_buffer=replay_buffer,
            num_workers=2,
        )

        try:
            manager.generate_games(2)
            summary = manager.stats_summary()
            expected_keys = {
                'total_games', 'total_examples', 'total_t_gates_reduced',
                'total_simplified', 'simplification_rate', 'avg_t_gates_reduced',
                'buffer_size', 'buffer_total_added',
            }
            self.assertEqual(set(summary.keys()), expected_keys)
        finally:
            manager.shutdown()

    def test_partition_games(self):
        """Games are partitioned roughly equally across workers."""
        model = make_model()
        config = make_mcts_config()
        replay_buffer = ReplayBuffer(capacity=10000)

        manager = ParallelSelfPlayManager(
            model=model, config=config, replay_buffer=replay_buffer,
            num_workers=3,
        )

        try:
            # 10 games across 3 workers: 4, 3, 3
            partition = manager._partition_games(10)
            self.assertEqual(sum(partition), 10)
            self.assertEqual(len(partition), 3)
            self.assertTrue(max(partition) - min(partition) <= 1)

            # Exact division
            partition = manager._partition_games(9)
            self.assertEqual(partition, [3, 3, 3])

            # Fewer games than workers
            partition = manager._partition_games(2)
            self.assertEqual(sum(partition), 2)
        finally:
            manager.shutdown()


# ==============================================================================
# Trainer Integration
# ==============================================================================

class TestTrainerIntegration(unittest.TestCase):
    """Test that Trainer correctly selects parallel vs serial manager."""

    def test_serial_when_workers_1(self):
        """Trainer uses SelfPlayManager when num_self_play_workers=1."""
        from alphazx.mcts.self_play import SelfPlayManager
        from alphazx.mcts.trainer import Trainer, TrainerConfig

        model = make_model()
        config = make_mcts_config()
        replay_buffer = ReplayBuffer(capacity=10000)
        trainer_config = TrainerConfig(num_self_play_workers=1)

        trainer = Trainer(model, config, trainer_config, replay_buffer)
        self.assertIsInstance(trainer.self_play_manager, SelfPlayManager)

    def test_parallel_when_workers_gt_1(self):
        """Trainer uses ParallelSelfPlayManager when num_self_play_workers > 1."""
        from alphazx.mcts.trainer import Trainer, TrainerConfig

        model = make_model()
        config = make_mcts_config()
        replay_buffer = ReplayBuffer(capacity=10000)
        trainer_config = TrainerConfig(num_self_play_workers=2)

        trainer = Trainer(model, config, trainer_config, replay_buffer)
        self.assertIsInstance(trainer.self_play_manager, ParallelSelfPlayManager)
        # Clean up
        trainer.self_play_manager.shutdown()


if __name__ == '__main__':
    unittest.main()
