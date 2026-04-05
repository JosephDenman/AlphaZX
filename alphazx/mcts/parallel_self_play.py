"""
Multi-process self-play for AlphaZero-style training.

Spawns N worker processes via ProcessPoolExecutor, each playing games
independently with a frozen copy of the model. Results are collected
in the main process and fed into the replay buffer.

This is the primary mechanism for scaling self-play throughput on
multi-core CPUs. Each worker is fully independent — no shared state,
no inter-process communication during game play.

Usage:
    manager = ParallelSelfPlayManager(model, config, replay_buffer, num_workers=4)
    results = manager.generate_games(100)  # 100 games across 4 workers
"""

from __future__ import annotations

import logging
import os
import random
import time
from concurrent.futures import ProcessPoolExecutor, Future
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from alphazx.mcts.config import MCTSConfig
from alphazx.mcts.replay_buffer import ReplayBuffer, TrainingExample
from alphazx.mcts.self_play import SelfPlayWorker, EpisodeResult, ACTION_TYPE_NAMES

logger = logging.getLogger(__name__)


def _build_model_from_hparams(hparams: dict) -> nn.Module:
    """Reconstruct an AlphaZXModel from hyperparameters dict.

    This import is deferred to avoid circular imports and to keep the
    worker function self-contained.
    """
    from alphazx.models.homogeneous.alphazx_model import AlphaZXModel
    return AlphaZXModel(
        num_node_types=hparams['num_node_types'],
        num_possible_phases=hparams['num_possible_phases'],
        num_possible_new_edges=hparams['num_possible_new_edges'],
        node_embedding_channels=hparams['node_embedding_channels'],
        num_edge_embeddings=hparams['num_edge_embeddings'],
        edge_embedding_channels=hparams['edge_embedding_channels'],
        pe_in_channels=hparams['pe_in_channels'],
        pe_out_channels=hparams['pe_out_channels'],
    )


def _extract_model_hparams(model: nn.Module) -> dict:
    """Extract the hyperparameters needed to reconstruct a model.

    Introspects the model's submodules to recover constructor arguments
    without requiring them to be stored explicitly.
    """
    rep = model.representation_network
    pred = model.prediction_network
    policy = pred.policy_network

    # From the FeatureEmbeddingLayer (rep.emb)
    emb_layer = rep.emb
    node_emb_channels = emb_layer.node_emb.embedding_dim
    edge_emb_channels = emb_layer.edge_emb.embedding_dim
    num_edge_embeddings = emb_layer.edge_emb.num_embeddings

    # PE dimensions from the Linear layer in FeatureEmbeddingLayer
    pe_in = emb_layer.pe_lin.in_features
    pe_out = emb_layer.pe_lin.out_features

    # From PolicyNetwork (stores these directly)
    num_node_types = policy.num_node_types
    num_possible_phases = policy.num_possible_phases
    num_possible_new_edges = policy.num_possible_new_edges

    return {
        'num_node_types': num_node_types,
        'num_possible_phases': num_possible_phases,
        'num_possible_new_edges': num_possible_new_edges,
        'node_embedding_channels': node_emb_channels,
        'num_edge_embeddings': num_edge_embeddings,
        'edge_embedding_channels': edge_emb_channels,
        'pe_in_channels': pe_in,
        'pe_out_channels': pe_out,
    }


def _worker_play_games(
    model_state_dict: dict,
    model_hparams: dict,
    mcts_config: MCTSConfig,
    num_games: int,
    worker_seed: int,
    difficulty_overrides: list[tuple[int, int]] | None = None,
) -> list[EpisodeResult]:
    """Play self-play games in a worker process.

    This is a module-level function (required for pickling by
    ProcessPoolExecutor). Each invocation:
    1. Seeds RNGs for reproducibility/diversity.
    2. Reconstructs the model and loads the state_dict.
    3. Creates a SelfPlayWorker and plays games sequentially.
    4. Returns the list of EpisodeResults.

    :param model_state_dict: Serialized model weights (dict of CPU tensors).
    :param model_hparams: Dict of constructor args for AlphaZXModel.
    :param mcts_config: MCTS configuration (will be temporarily mutated
                        if difficulty_overrides is provided).
    :param num_games: Number of games this worker should play.
    :param worker_seed: Seed for this worker's RNGs.
    :param difficulty_overrides: Optional list of (num_qubits, depth) tuples,
                                 one per game, for curriculum support.
    :return: List of EpisodeResult objects.
    """
    # --- Prevent thread oversubscription ---
    # Each worker is a separate process. By default PyTorch (via OpenMP/MKL)
    # will spawn as many threads as there are CPU cores *per process*.
    # With N workers this means N × num_cores threads competing for the same
    # cores, causing massive contention and apparent hangs on many-core
    # machines (e.g. Mac Studio).  Pin each worker to a single thread.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'  # macOS Accelerate
    torch.set_num_threads(1)

    # Seed all RNGs so different workers produce different games
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))

    # Suppress per-game INFO logging in workers to avoid interleaved output
    logging.getLogger('alphazx').setLevel(logging.WARNING)

    # Reconstruct model and load weights
    model = _build_model_from_hparams(model_hparams)
    model.load_state_dict(model_state_dict)
    model.eval()

    worker = SelfPlayWorker(model, mcts_config, device=torch.device('cpu'))
    results: list[EpisodeResult] = []

    for i in range(num_games):
        # Apply curriculum difficulty override if provided
        if difficulty_overrides and i < len(difficulty_overrides):
            saved_q = mcts_config.num_qubits
            saved_d = mcts_config.depth
            mcts_config.num_qubits, mcts_config.depth = difficulty_overrides[i]

        try:
            result = worker.play_episode()
            results.append(result)
        except Exception as e:
            # Log but don't crash the worker — skip this game
            logging.getLogger(__name__).warning(
                f"Worker (seed={worker_seed}) game {i} failed: {e}"
            )
        finally:
            # Restore config if we overrode it
            if difficulty_overrides and i < len(difficulty_overrides):
                mcts_config.num_qubits = saved_q
                mcts_config.depth = saved_d

    return results


class ParallelSelfPlayManager:
    """Orchestrates multi-process self-play game generation.

    Drop-in replacement for SelfPlayManager with the same generate_games()
    interface. Internally spawns worker processes that each play a share
    of the requested games, then collects results and inserts examples
    into the replay buffer.

    The ProcessPoolExecutor is created once at construction time and
    reused across iterations to avoid repeated process startup costs.
    """

    def __init__(
        self,
        model: nn.Module,
        config: MCTSConfig,
        replay_buffer: ReplayBuffer,
        device: torch.device = torch.device('cpu'),
        num_workers: int = 4,
    ):
        self.model = model
        self.config = config
        self.replay_buffer = replay_buffer
        self.device = device
        self.num_workers = num_workers

        # Extract model hyperparameters once for worker reconstruction
        self._model_hparams = _extract_model_hparams(model)

        # Set thread-limiting env vars *before* creating the pool so that
        # spawn-ed children inherit them before importing PyTorch/NumPy.
        if num_workers > 1:
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

        # Create the process pool using the platform default start method.
        # - Linux: defaults to 'fork' (fast, copy-on-write)
        # - macOS: defaults to 'spawn' (required — 'fork' deadlocks with
        #   the Obj-C runtime and macOS system libraries since Python 3.8)
        # We do NOT override the start method; the platform default is the
        # only safe choice on each OS.
        self._executor = ProcessPoolExecutor(
            max_workers=num_workers,
        )

        # Lifetime statistics (mirrors SelfPlayManager interface)
        self.total_games: int = 0
        self.total_examples: int = 0
        self.total_t_gates_reduced: int = 0
        self.total_simplified: int = 0

    def generate_games(
        self,
        num_games: int,
        start_diagrams: Optional[list] = None,
        difficulty_overrides: list[tuple[int, int]] | None = None,
    ) -> list[EpisodeResult]:
        """Generate self-play games across multiple worker processes.

        :param num_games: Total number of games to play.
        :param start_diagrams: Not supported in parallel mode (ignored with warning).
                               Use difficulty_overrides for curriculum support.
        :param difficulty_overrides: Optional list of (num_qubits, depth) tuples,
                                     one per game. Partitioned across workers.
        :return: List of EpisodeResult summaries from all workers.
        """
        if start_diagrams is not None:
            logger.warning(
                "ParallelSelfPlayManager does not support start_diagrams "
                "(ZXDiagram objects may not be safely picklable across all "
                "configurations). Ignoring start_diagrams; workers will "
                "generate random circuits."
            )

        # Serialize model weights once for this iteration
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

        # Partition games across workers
        games_per_worker = self._partition_games(num_games)

        # Partition difficulty overrides if provided
        override_partitions = self._partition_overrides(
            difficulty_overrides, games_per_worker,
        )

        # Generate per-worker seeds
        base_seed = int(time.time() * 1000) % (2**31)
        worker_seeds = [base_seed + i * 10_000 for i in range(self.num_workers)]

        # Dispatch to workers
        logger.info(
            f"Dispatching {num_games} games across {self.num_workers} workers "
            f"({games_per_worker})..."
        )
        t_start = time.time()
        futures: list[Future] = []
        for i, n_games in enumerate(games_per_worker):
            if n_games == 0:
                continue
            future = self._executor.submit(
                _worker_play_games,
                model_state_dict=state_dict,
                model_hparams=self._model_hparams,
                mcts_config=self.config,
                num_games=n_games,
                worker_seed=worker_seeds[i],
                difficulty_overrides=override_partitions[i] if override_partitions else None,
            )
            futures.append(future)

        # Collect results from all workers
        logger.info(f"All {len(futures)} workers dispatched. Waiting for results...")
        all_results: list[EpisodeResult] = []
        for idx, future in enumerate(futures):
            try:
                worker_results = future.result()
                all_results.extend(worker_results)
                logger.info(
                    f"  Worker {idx + 1}/{len(futures)} finished: "
                    f"{len(worker_results)} games"
                )
            except Exception as e:
                logger.error(
                    f"Worker {idx + 1}/{len(futures)} failed: {e}. "
                    f"Skipping its games for this iteration."
                )

        t_elapsed = time.time() - t_start

        # Insert examples into replay buffer (sequential — buffer is not thread-safe)
        for result in all_results:
            self.replay_buffer.add_game(result.examples)

            # Update lifetime statistics
            self.total_games += 1
            self.total_examples += len(result.examples)
            self.total_t_gates_reduced += result.t_gates_reduced
            if result.simplified:
                self.total_simplified += 1

        # Log aggregate summary
        n = max(1, len(all_results))
        games_per_sec = len(all_results) / max(t_elapsed, 0.001)
        logger.info(
            f"Parallel self-play: {len(all_results)} games in {t_elapsed:.1f}s "
            f"across {self.num_workers} workers ({games_per_sec:.1f} games/s), "
            f"avg_steps={sum(r.num_steps for r in all_results) / n:.1f}, "
            f"avg_t_reduced={sum(r.t_gates_reduced for r in all_results) / n:.1f}"
        )

        return all_results

    def _partition_games(self, num_games: int) -> list[int]:
        """Partition num_games roughly equally across workers.

        Returns a list of length num_workers with the number of games
        each worker should play. Remainder games go to the first workers.
        """
        base = num_games // self.num_workers
        remainder = num_games % self.num_workers
        return [
            base + (1 if i < remainder else 0)
            for i in range(self.num_workers)
        ]

    def _partition_overrides(
        self,
        overrides: list[tuple[int, int]] | None,
        games_per_worker: list[int],
    ) -> list[list[tuple[int, int]]] | None:
        """Partition difficulty overrides to match game partitioning."""
        if overrides is None:
            return None

        partitions = []
        offset = 0
        for n_games in games_per_worker:
            partitions.append(overrides[offset:offset + n_games])
            offset += n_games
        return partitions

    def stats_summary(self) -> dict:
        """Return a summary of self-play statistics (mirrors SelfPlayManager)."""
        return {
            'total_games': self.total_games,
            'total_examples': self.total_examples,
            'total_t_gates_reduced': self.total_t_gates_reduced,
            'total_simplified': self.total_simplified,
            'simplification_rate': (
                self.total_simplified / max(1, self.total_games)
            ),
            'avg_t_gates_reduced': (
                self.total_t_gates_reduced / max(1, self.total_games)
            ),
            'buffer_size': len(self.replay_buffer),
            'buffer_total_added': self.replay_buffer.total_added,
        }

    def shutdown(self) -> None:
        """Shutdown the process pool. Called automatically if using context manager."""
        self._executor.shutdown(wait=True)

    def __del__(self):
        """Best-effort cleanup of the process pool."""
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
