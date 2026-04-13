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
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, Future
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from alphazx.mcts.config import MCTSConfig
from alphazx.mcts.replay_buffer import ReplayBuffer, TrainingExample
from alphazx.mcts.self_play import SelfPlayWorker, MultiGameSelfPlayWorker, EpisodeResult, ACTION_TYPE_NAMES

logger = logging.getLogger(__name__)


def _build_model_from_hparams(hparams: dict) -> nn.Module:
    """Reconstruct a model from hyperparameters dict.

    Supports both AlphaZXModel (homogeneous) and AlphaZXHeteroModel
    (heterogeneous). The 'model_type' key determines which to build.

    This import is deferred to avoid circular imports and to keep the
    worker function self-contained.
    """
    model_type = hparams.get('model_type', 'homogeneous')

    if model_type == 'heterogeneous':
        from alphazx.models.heterogeneous.alphazx_hetero_model import AlphaZXHeteroModel
        # Pass through HGT-specific hyperparameters when available; fall
        # back to AlphaZXHeteroModel defaults for older hparams dicts.
        hgt_kwargs = {}
        for key in (
            'hgt_num_shared_layers', 'hgt_num_policy_layers',
            'hgt_num_value_layers', 'hgt_heads', 'hgt_dropout',
        ):
            if key in hparams:
                hgt_kwargs[key] = hparams[key]
        return AlphaZXHeteroModel(
            num_node_types=hparams['num_node_types'],
            num_possible_phases=hparams['num_possible_phases'],
            num_possible_new_edges=hparams['num_possible_new_edges'],
            node_embedding_channels=hparams['node_embedding_channels'],
            num_edge_embeddings=hparams['num_edge_embeddings'],
            edge_embedding_channels=hparams['edge_embedding_channels'],
            pe_in_channels=hparams['pe_in_channels'],
            pe_out_channels=hparams['pe_out_channels'],
            **hgt_kwargs,
        )
    else:
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


def _unwrap_compiled(model: nn.Module) -> nn.Module:
    """Unwrap a torch.compile'd model to the original nn.Module.

    torch.compile wraps the model in an OptimizedModule.  Most attribute
    access is delegated, but isinstance() checks fail.  This helper
    returns the underlying module so introspection works correctly.
    """
    orig = getattr(model, '_orig_mod', None)
    return orig if orig is not None else model


def _extract_model_hparams(model: nn.Module) -> dict:
    """Extract the hyperparameters needed to reconstruct a model.

    Introspects the model's submodules to recover constructor arguments
    without requiring them to be stored explicitly. Supports both
    AlphaZXModel (homogeneous) and AlphaZXHeteroModel (heterogeneous).

    Handles torch.compile'd models by unwrapping to the original module.
    """
    from alphazx.models.heterogeneous.alphazx_hetero_model import AlphaZXHeteroModel

    model = _unwrap_compiled(model)
    is_hetero = isinstance(model, AlphaZXHeteroModel)

    # Both model types have a FeatureEmbeddingLayer named 'emb'
    if is_hetero:
        emb_layer = model.emb
        # Hetero model stores these directly
        num_node_types = model.num_node_types
        num_possible_phases = model.num_possible_phases
        num_possible_new_edges = model.num_possible_new_edges
    else:
        rep = model.representation_network
        emb_layer = rep.emb
        pred = model.prediction_network
        policy = pred.policy_network
        num_node_types = policy.num_node_types
        num_possible_phases = policy.num_possible_phases
        num_possible_new_edges = policy.num_possible_new_edges

    node_emb_channels = emb_layer.node_emb.embedding_dim
    edge_emb_channels = emb_layer.edge_emb.embedding_dim
    num_edge_embeddings = emb_layer.edge_emb.num_embeddings
    pe_in = emb_layer.pe_lin.in_features
    pe_out = emb_layer.pe_lin.out_features

    hparams = {
        'num_node_types': num_node_types,
        'num_possible_phases': num_possible_phases,
        'num_possible_new_edges': num_possible_new_edges,
        'node_embedding_channels': node_emb_channels,
        'num_edge_embeddings': num_edge_embeddings,
        'edge_embedding_channels': edge_emb_channels,
        'pe_in_channels': pe_in,
        'pe_out_channels': pe_out,
        'model_type': 'heterogeneous' if is_hetero else 'homogeneous',
    }

    # HGT-specific hyperparameters (layer counts, heads, dropout) — needed
    # so the worker reconstructs the model with the correct architecture.
    if is_hetero:
        hparams['hgt_num_shared_layers'] = len(model.shared_hgt)
        hparams['hgt_num_policy_layers'] = len(model.policy_hgt)
        hparams['hgt_num_value_layers'] = len(model.value_hgt)
        # heads and dropout live on individual HGTBlocks; read from the first
        first_block = model.shared_hgt[0]
        hparams['hgt_heads'] = first_block.conv.heads
        hparams['hgt_dropout'] = first_block.dropout.p

    return hparams


def _worker_play_games(
    model_state_dict: dict,
    model_hparams: dict,
    mcts_config: MCTSConfig,
    num_games: int,
    worker_seed: int,
    difficulty_overrides: list[tuple[int, int]] | None = None,
    concurrent_games: int = 1,
) -> list[EpisodeResult]:
    """Play self-play games in a worker process.

    This is a module-level function (required for pickling by
    ProcessPoolExecutor). Each invocation:
    1. Seeds RNGs for reproducibility/diversity.
    2. Reconstructs the model and loads the state_dict.
    3. Creates a SelfPlayWorker (or MultiGameSelfPlayWorker) and plays games.
    4. Returns the list of EpisodeResults.

    :param model_state_dict: Serialized model weights (dict of CPU tensors).
    :param model_hparams: Dict of constructor args for AlphaZXModel.
    :param mcts_config: MCTS configuration (will be temporarily mutated
                        if difficulty_overrides is provided).
    :param num_games: Number of games this worker should play.
    :param worker_seed: Seed for this worker's RNGs.
    :param difficulty_overrides: Optional list of (num_qubits, depth) tuples,
                                 one per game, for curriculum support.
    :param concurrent_games: Number of games to interleave within this worker.
                             When > 1, uses MultiGameSelfPlayWorker for cross-game
                             batched MCTS. Default 1 = sequential (original behavior).
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

    # Keep most worker logging at WARNING but allow progress messages
    # from the worker logger and MultiGameSelfPlayWorker to stderr.
    logging.getLogger('alphazx').setLevel(logging.WARNING)

    _stderr_fmt = logging.Formatter(
        '%(asctime)s [worker-%(process)d] %(message)s', datefmt='%H:%M:%S',
    )

    _wlog = logging.getLogger(f'{__name__}.worker')
    _wlog.setLevel(logging.INFO)
    if not _wlog.handlers:
        _h = logging.StreamHandler(sys.stderr)
        _h.setFormatter(_stderr_fmt)
        _wlog.addHandler(_h)

    # Allow MultiGameSelfPlayWorker's per-game progress logs through
    _sp_log = logging.getLogger('alphazx.mcts.self_play')
    _sp_log.setLevel(logging.INFO)
    if not _sp_log.handlers:
        _h2 = logging.StreamHandler(sys.stderr)
        _h2.setFormatter(_stderr_fmt)
        _sp_log.addHandler(_h2)

    t_worker_start = time.time()

    # Reconstruct model and load weights
    model = _build_model_from_hparams(model_hparams)
    model.load_state_dict(model_state_dict)
    model.eval()

    # Optional torch.compile for faster inference
    if mcts_config.torch_compile:
        try:
            model = torch.compile(model, dynamic=True)
            _wlog.info("torch.compile applied (compilation on first forward pass)")
        except Exception as e:
            _wlog.warning(f"torch.compile failed, using eager mode: {e}")

    # --- Cross-game batched path ---
    if concurrent_games > 1:
        _wlog.info(
            f"Starting: PID={os.getpid()}, {num_games} games, "
            f"concurrent_games={concurrent_games}"
        )
        multi_worker = MultiGameSelfPlayWorker(
            model, mcts_config, device=torch.device('cpu'),
            concurrent_games=concurrent_games,
        )
        try:
            results = multi_worker.play_episodes(num_games, difficulty_overrides)
        except Exception as e:
            _wlog.error(f"MultiGameSelfPlayWorker failed: {e}")
            results = []

        _wlog.info(
            f"Worker finished: {len(results)}/{num_games} games in "
            f"{time.time() - t_worker_start:.1f}s (concurrent_games={concurrent_games})"
        )
        return results

    # --- Sequential path (original behavior) ---
    worker = SelfPlayWorker(model, mcts_config, device=torch.device('cpu'))
    results: list[EpisodeResult] = []

    for i in range(num_games):
        t_game = time.time()

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
            _wlog.warning(
                f"Game {i+1}/{num_games} failed ({time.time() - t_game:.1f}s): {e}"
            )
        finally:
            # Restore config if we overrode it
            if difficulty_overrides and i < len(difficulty_overrides):
                mcts_config.num_qubits = saved_q
                mcts_config.depth = saved_d

    _wlog.info(
        f"Worker finished: {len(results)}/{num_games} games in "
        f"{time.time() - t_worker_start:.1f}s"
    )
    return results


class PendingSelfPlay:
    """Handle for an in-flight batch of self-play games.

    Returned by ``ParallelSelfPlayManager.dispatch_games()`` so the caller
    can do other work (e.g. training) while workers are still playing.

    Call :meth:`collect` to block until all workers finish and return the
    aggregated results.  Results are **not** automatically inserted into
    the replay buffer — the caller must do that (the Trainer handles it).
    """

    def __init__(
        self,
        futures: list[Future],
        num_games: int,
        num_workers: int,
        t_start: float,
    ):
        self._futures = futures
        self.num_games = num_games
        self.num_workers = num_workers
        self._t_start = t_start
        self._collected = False

    @property
    def done(self) -> bool:
        """True if all worker futures have completed (non-blocking check)."""
        return all(f.done() for f in self._futures)

    def collect(self) -> list[EpisodeResult]:
        """Block until all workers finish and return results.

        Can only be called once.  Subsequent calls raise RuntimeError.
        """
        if self._collected:
            raise RuntimeError("PendingSelfPlay.collect() already called")
        self._collected = True

        all_results: list[EpisodeResult] = []
        for idx, future in enumerate(self._futures):
            try:
                worker_results = future.result()
                all_results.extend(worker_results)
                logger.info(
                    f"  Worker {idx + 1}/{len(self._futures)} finished: "
                    f"{len(worker_results)} games"
                )
            except Exception as e:
                logger.error(
                    f"Worker {idx + 1}/{len(self._futures)} failed: {e}. "
                    f"Skipping its games for this iteration."
                )

        t_elapsed = time.time() - self._t_start
        n = max(1, len(all_results))
        games_per_sec = len(all_results) / max(t_elapsed, 0.001)
        logger.info(
            f"Parallel self-play: {len(all_results)} games in {t_elapsed:.1f}s "
            f"across {self.num_workers} workers ({games_per_sec:.1f} games/s), "
            f"avg_steps={sum(r.num_steps for r in all_results) / n:.1f}, "
            f"avg_t_reduced={sum(r.t_gates_reduced for r in all_results) / n:.1f}"
        )

        return all_results


class ParallelSelfPlayManager:
    """Orchestrates multi-process self-play game generation.

    Drop-in replacement for SelfPlayManager with the same generate_games()
    interface. Internally spawns worker processes that each play a share
    of the requested games, then collects results and inserts examples
    into the replay buffer.

    The ProcessPoolExecutor is created once at construction time and
    reused across iterations to avoid repeated process startup costs.

    Supports two modes of operation:

    **Blocking** (original interface)::

        results = manager.generate_games(100)

    **Pipelined** (for overlapping self-play with training)::

        pending = manager.dispatch_games(100)
        # ... do other work while workers play ...
        results = pending.collect()
        manager.ingest_results(results)
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

        # Explicitly use 'spawn' context for the process pool.
        # - 'fork' deadlocks on macOS (Obj-C runtime, system libraries,
        #   and PyTorch's threading are all incompatible with fork).
        # - macOS *should* default to 'spawn' since Python 3.8, but some
        #   environments (conda, certain Python builds) can revert to 'fork'.
        # - 'spawn' also works correctly on Linux (just slower startup).
        # By being explicit we avoid environment-dependent hangs.
        self._executor = ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context('spawn'),
        )

        # Lifetime statistics (mirrors SelfPlayManager interface)
        self.total_games: int = 0
        self.total_examples: int = 0
        self.total_t_gates_reduced: int = 0
        self.total_simplified: int = 0

    def dispatch_games(
        self,
        num_games: int,
        difficulty_overrides: list[tuple[int, int]] | None = None,
    ) -> PendingSelfPlay:
        """Dispatch self-play games to workers (non-blocking).

        Returns a :class:`PendingSelfPlay` handle.  The caller can do other
        work while the workers play, then call ``handle.collect()`` to
        block until all results are ready.

        :param num_games: Total number of games to play.
        :param difficulty_overrides: Optional list of (num_qubits, depth) tuples,
                                     one per game. Partitioned across workers.
        :return: A PendingSelfPlay handle for later collection.
        """
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
                concurrent_games=self.config.concurrent_games,
            )
            futures.append(future)

        logger.info(f"All {len(futures)} workers dispatched.")

        return PendingSelfPlay(
            futures=futures,
            num_games=num_games,
            num_workers=self.num_workers,
            t_start=t_start,
        )

    def ingest_results(self, results: list[EpisodeResult]) -> None:
        """Insert self-play results into the replay buffer and update stats.

        Called by the Trainer after ``pending.collect()`` returns.
        Separated from dispatch/collect so the Trainer controls exactly
        when buffer insertion happens.
        """
        for result in results:
            self.replay_buffer.add_game(result.examples)

            # Update lifetime statistics
            self.total_games += 1
            self.total_examples += len(result.examples)
            self.total_t_gates_reduced += result.t_gates_reduced
            if result.simplified:
                self.total_simplified += 1

    def generate_games(
        self,
        num_games: int,
        start_diagrams: Optional[list] = None,
        difficulty_overrides: list[tuple[int, int]] | None = None,
    ) -> list[EpisodeResult]:
        """Generate self-play games across multiple worker processes (blocking).

        This is the original blocking interface.  Equivalent to calling
        ``dispatch_games()`` followed by ``collect()`` and ``ingest_results()``.

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

        pending = self.dispatch_games(num_games, difficulty_overrides)
        results = pending.collect()
        self.ingest_results(results)
        return results

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
