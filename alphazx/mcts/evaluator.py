"""
Evaluator for comparing MCTS-guided policy against baselines.

The primary evaluation metric is T-gate reduction: how many non-Clifford gates
does the agent remove compared to the initial circuit, and how does this compare
to PyZX's built-in simplification routines.

Evaluation uses low temperature (near-greedy) MCTS to select the best actions.

Supports parallel evaluation via ParallelEvaluator, which distributes eval games
across worker processes the same way ParallelSelfPlayManager distributes self-play.
"""

from __future__ import annotations

import copy
import logging
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor, Future
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pyzx
import torch
import torch.nn as nn

from alphazx.diagram.diagram_generators import (
    clifford_zx_diagram, cnot_had_phase_zx_diagram,
    clifford_zx_diagram_with_pyzx, cnot_had_phase_zx_diagram_with_pyzx,
)
from alphazx.game.zx_game import num_non_clifford_gates
from alphazx.mcts.config import MCTSConfig
from alphazx.mcts.game_state import GameState
from alphazx.mcts.search import MCTS

logger = logging.getLogger(__name__)


@dataclass
class EvalGameResult:
    """Result from a single evaluation game."""
    initial_t_gates: int
    final_t_gates: int
    t_gates_reduced: int
    num_steps: int
    simplified: bool
    pyzx_t_gates: Optional[int]  # T-gate count after PyZX full_reduce
    pyzx_t_gates_reduced: Optional[int]


@dataclass
class EvalSummary:
    """Aggregated evaluation results."""
    num_games: int
    avg_t_gates_reduced: float
    avg_reduction_ratio: float  # fraction of initial T-gates removed
    simplification_rate: float  # fraction of games fully simplified
    avg_steps: float
    # PyZX comparison
    pyzx_avg_t_gates_reduced: Optional[float]
    pyzx_avg_reduction_ratio: Optional[float]
    # How often the agent matches or beats PyZX
    agent_vs_pyzx_wins: int
    agent_vs_pyzx_ties: int
    agent_vs_pyzx_losses: int
    wall_time: float

    def __repr__(self) -> str:
        s = (
            f"EvalSummary(games={self.num_games}, "
            f"agent_reduced={self.avg_t_gates_reduced:.2f} ({self.avg_reduction_ratio:.1%}), "
            f"simplified={self.simplification_rate:.1%}, "
            f"steps={self.avg_steps:.1f}"
        )
        if self.pyzx_avg_t_gates_reduced is not None:
            s += (
                f", pyzx_reduced={self.pyzx_avg_t_gates_reduced:.2f} "
                f"({self.pyzx_avg_reduction_ratio:.1%}), "
                f"vs_pyzx: W{self.agent_vs_pyzx_wins}/T{self.agent_vs_pyzx_ties}/L{self.agent_vs_pyzx_losses}"
            )
        s += f", time={self.wall_time:.1f}s)"
        return s


class Evaluator:
    """Evaluates the trained model against baselines.

    Runs near-greedy MCTS (low temperature) on a set of test circuits
    and compares the T-gate reduction to PyZX's built-in simplification.
    """

    def __init__(
        self,
        mcts_config: MCTSConfig,
        device: torch.device = torch.device('cpu'),
        compare_pyzx: bool = True,
        eval_temperature: float = 0.1,
        max_eval_steps: Optional[int] = None,
        game_timeout: float = 0.0,
    ):
        """
        :param mcts_config: MCTS configuration. Temperature will be overridden
                            for evaluation with eval_temperature.
        :param device: Device for neural network inference.
        :param compare_pyzx: Whether to also run PyZX simplification for comparison.
        :param eval_temperature: Temperature for action selection during evaluation.
                                 Low values (0.1) give near-greedy behavior.
        :param max_eval_steps: Maximum steps per eval game. If None, uses
                               mcts_config.max_episode_length. Use a lower value
                               (e.g. 50) to prevent the agent from wasting compute
                               on hopeless games early in training.
        :param game_timeout: Per-game wall-clock timeout in seconds. 0 = no timeout.
                             If a game exceeds this, it is terminated early and the
                             result so far is recorded.
        """
        self.base_config = mcts_config
        self.device = device
        self.compare_pyzx = compare_pyzx
        self.eval_temperature = eval_temperature
        self.max_eval_steps = max_eval_steps
        self.game_timeout = game_timeout

    def evaluate(
        self,
        model: nn.Module,
        num_games: int = 20,
        num_qubits: Optional[int] = None,
        depth: Optional[int] = None,
        fixed_circuits: Optional[list] = None,
    ) -> EvalSummary:
        """Run evaluation games and return aggregated results.

        :param model: The trained model to evaluate.
        :param num_games: Number of evaluation games to play.
        :param num_qubits: Override circuit size (defaults to mcts_config).
        :param depth: Override circuit depth (defaults to mcts_config).
        :param fixed_circuits: Optional list of (ZXDiagram, pyzx_graph) tuples.
                               If provided, num_games is ignored.
        :return: EvalSummary with aggregated statistics.
        """
        start_time = time.time()

        # Build eval config with low temperature
        eval_config = copy.copy(self.base_config)
        eval_config.temperature = self.eval_temperature

        nq = num_qubits or self.base_config.num_qubits
        d = depth or self.base_config.depth

        mcts = MCTS(model, eval_config)
        was_training = model.training
        model.eval()

        results: list[EvalGameResult] = []

        try:
            if fixed_circuits is not None:
                # fixed_circuits can be list of (ZXDiagram, pyzx_graph) tuples
                # or list of ZXDiagrams (backward compat)
                circuits_to_eval = fixed_circuits
            else:
                # Pre-generate all eval circuits so both agent and PyZX
                # use the exact same circuit for apples-to-apples comparison.
                circuits_to_eval = []
                for _ in range(num_games):
                    diagram, pyzx_graph = self._generate_eval_circuit(nq, d)
                    circuits_to_eval.append((diagram, pyzx_graph))

            for i, item in enumerate(circuits_to_eval):
                # Handle both tuple (diagram, pyzx_graph) and bare diagram
                if isinstance(item, tuple):
                    diagram, pyzx_graph = item
                else:
                    diagram, pyzx_graph = item, None

                result = self._play_eval_game(mcts, diagram, pyzx_graph)
                results.append(result)
                logger.debug(
                    f"Eval game {i+1}: "
                    f"t_gates {result.initial_t_gates}→{result.final_t_gates} "
                    f"(-{result.t_gates_reduced}), "
                    f"pyzx: -{result.pyzx_t_gates_reduced}"
                )
        finally:
            # Restore model mode to what it was before evaluation
            if was_training:
                model.train()

        wall_time = time.time() - start_time
        return self._aggregate_results(results, wall_time)

    def _generate_eval_circuit(
        self, num_qubits: int, depth: int,
    ) -> tuple:
        """Generate a circuit and return both ZXDiagram and PyZX graph."""
        if self.base_config.circuit_type == 'cnot_had_phase':
            return cnot_had_phase_zx_diagram_with_pyzx(
                num_qubits, depth,
                self.base_config.p_had, self.base_config.p_t,
            )
        else:
            return clifford_zx_diagram_with_pyzx(num_qubits, depth, t_gates=True)

    def _play_eval_game(
        self,
        mcts: MCTS,
        diagram,
        pyzx_graph=None,
    ) -> EvalGameResult:
        """Play a single evaluation game with near-greedy MCTS.

        Both the agent and PyZX baseline operate on the SAME circuit for
        a fair apples-to-apples comparison.

        Respects max_eval_steps (caps the number of MCTS steps) and
        game_timeout (caps wall-clock time per game).
        """
        state = GameState.from_diagram(diagram.copy())
        initial_t_gates = state.num_non_clifford

        # Run PyZX simplification on the same circuit for comparison
        pyzx_t_gates = None
        pyzx_reduced = None
        if self.compare_pyzx and pyzx_graph is not None:
            pyzx_t_gates, pyzx_reduced = self._run_pyzx_baseline(pyzx_graph)

        # Determine the step limit for this eval game
        max_steps = self.max_eval_steps or mcts.config.max_episode_length

        # Play the game with MCTS
        num_steps = 0
        game_start = time.time()
        timed_out = False
        while num_steps < max_steps:
            if state.is_terminal() or not state.has_legal_actions():
                break

            # Check wall-clock timeout
            if self.game_timeout > 0 and (time.time() - game_start) > self.game_timeout:
                timed_out = True
                break

            action, policy, _ = mcts.select_action(state, self.device)
            if not policy:
                break

            try:
                _, done = state.apply_action(action)
            except (ValueError, KeyError, IndexError, AssertionError):
                break

            num_steps += 1
            if done:
                break

        final_t_gates = state.num_non_clifford
        t_gates_reduced = initial_t_gates - final_t_gates

        if timed_out:
            logger.debug(
                f"Eval game timed out after {num_steps} steps "
                f"({time.time() - game_start:.0f}s)"
            )

        return EvalGameResult(
            initial_t_gates=initial_t_gates,
            final_t_gates=final_t_gates,
            t_gates_reduced=t_gates_reduced,
            num_steps=num_steps,
            simplified=state.is_terminal(),
            pyzx_t_gates=pyzx_t_gates,
            pyzx_t_gates_reduced=pyzx_reduced,
        )

    def _run_pyzx_baseline(
        self,
        pyzx_graph,
    ) -> tuple[int, int]:
        """Run PyZX's full_reduce on the SAME circuit the agent evaluated.

        This now takes the original PyZX graph (before simplification) and runs
        full_reduce on a copy, giving a true apples-to-apples comparison.
        """
        try:
            initial_t = pyzx.tcount(pyzx_graph)
            reduced_graph = pyzx_graph.copy()
            pyzx.full_reduce(reduced_graph)
            final_t = pyzx.tcount(reduced_graph)
            return final_t, initial_t - final_t
        except Exception as e:
            logger.warning(f"PyZX baseline failed: {e}")
            return None, None

    def _aggregate_results(
        self,
        results: list[EvalGameResult],
        wall_time: float,
    ) -> EvalSummary:
        """Aggregate per-game results into a summary."""
        n = len(results)
        if n == 0:
            return EvalSummary(
                num_games=0,
                avg_t_gates_reduced=0, avg_reduction_ratio=0,
                simplification_rate=0, avg_steps=0,
                pyzx_avg_t_gates_reduced=None, pyzx_avg_reduction_ratio=None,
                agent_vs_pyzx_wins=0, agent_vs_pyzx_ties=0, agent_vs_pyzx_losses=0,
                wall_time=wall_time,
            )

        total_reduced = sum(r.t_gates_reduced for r in results)
        total_initial = sum(r.initial_t_gates for r in results)
        total_steps = sum(r.num_steps for r in results)
        total_simplified = sum(1 for r in results if r.simplified)

        avg_reduction_ratio = total_reduced / max(1, total_initial)

        # PyZX comparison
        pyzx_results = [r for r in results if r.pyzx_t_gates_reduced is not None]
        pyzx_avg_reduced = None
        pyzx_avg_ratio = None
        wins = ties = losses = 0

        if pyzx_results:
            pyzx_total_reduced = sum(r.pyzx_t_gates_reduced for r in pyzx_results)
            pyzx_total_initial = sum(r.initial_t_gates for r in pyzx_results)
            pyzx_avg_reduced = pyzx_total_reduced / len(pyzx_results)
            pyzx_avg_ratio = pyzx_total_reduced / max(1, pyzx_total_initial)

            for r in pyzx_results:
                if r.t_gates_reduced > r.pyzx_t_gates_reduced:
                    wins += 1
                elif r.t_gates_reduced == r.pyzx_t_gates_reduced:
                    ties += 1
                else:
                    losses += 1

        return EvalSummary(
            num_games=n,
            avg_t_gates_reduced=total_reduced / n,
            avg_reduction_ratio=avg_reduction_ratio,
            simplification_rate=total_simplified / n,
            avg_steps=total_steps / n,
            pyzx_avg_t_gates_reduced=pyzx_avg_reduced,
            pyzx_avg_reduction_ratio=pyzx_avg_ratio,
            agent_vs_pyzx_wins=wins,
            agent_vs_pyzx_ties=ties,
            agent_vs_pyzx_losses=losses,
            wall_time=wall_time,
        )


# ---------------------------------------------------------------------------
# Parallel evaluation
# ---------------------------------------------------------------------------

def _build_model_from_hparams(hparams: dict) -> nn.Module:
    """Reconstruct a model from hyperparameters dict.

    Supports both AlphaZXModel (homogeneous) and AlphaZXHeteroModel.
    Same logic as parallel_self_play._build_model_from_hparams.
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


def _eval_worker_play_games(
    model_state_dict: dict,
    model_hparams: dict,
    mcts_config: MCTSConfig,
    num_games: int,
    worker_seed: int,
    eval_temperature: float,
    compare_pyzx: bool,
    max_eval_steps: Optional[int],
    game_timeout: float,
    num_qubits: int,
    depth: int,
) -> list[EvalGameResult]:
    """Play evaluation games in a worker process.

    Module-level function required for pickling by ProcessPoolExecutor.
    Each invocation:
    1. Seeds RNGs for reproducibility.
    2. Reconstructs the model, loads weights, sets eval mode.
    3. Creates an Evaluator and plays games sequentially within this worker.
    4. Returns the list of EvalGameResult objects.
    """
    # --- Prevent thread oversubscription (same as parallel_self_play) ---
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    torch.set_num_threads(1)

    # Seed all RNGs for diverse but reproducible eval circuits
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))

    # Worker diagnostic logging
    logging.getLogger('alphazx').setLevel(logging.WARNING)
    _wlog = logging.getLogger(f'{__name__}.eval_worker')
    _wlog.setLevel(logging.INFO)
    if not _wlog.handlers:
        _h = logging.StreamHandler(sys.stderr)
        _h.setFormatter(logging.Formatter(
            '%(asctime)s [eval-worker-%(process)d] %(message)s', datefmt='%H:%M:%S',
        ))
        _wlog.addHandler(_h)

    t_worker_start = time.time()

    # Reconstruct model and load weights
    model = _build_model_from_hparams(model_hparams)
    model.load_state_dict(model_state_dict)
    model.eval()

    # Build eval-specific MCTS config
    eval_config = copy.copy(mcts_config)
    eval_config.temperature = eval_temperature

    mcts = MCTS(model, eval_config)

    # Determine step limit
    max_steps = max_eval_steps or mcts_config.max_episode_length

    results: list[EvalGameResult] = []
    for _ in range(num_games):
        try:
            # Generate circuit
            if mcts_config.circuit_type == 'cnot_had_phase':
                diagram, pyzx_graph = cnot_had_phase_zx_diagram_with_pyzx(
                    num_qubits, depth, mcts_config.p_had, mcts_config.p_t,
                )
            else:
                diagram, pyzx_graph = clifford_zx_diagram_with_pyzx(
                    num_qubits, depth, t_gates=True,
                )

            # Run PyZX baseline
            pyzx_t_gates = None
            pyzx_reduced = None
            if compare_pyzx and pyzx_graph is not None:
                try:
                    initial_t = pyzx.tcount(pyzx_graph)
                    reduced_graph = pyzx_graph.copy()
                    pyzx.full_reduce(reduced_graph)
                    final_t = pyzx.tcount(reduced_graph)
                    pyzx_t_gates = final_t
                    pyzx_reduced = initial_t - final_t
                except Exception:
                    pass

            # Play eval game
            state = GameState.from_diagram(diagram.copy())
            initial_t_gates = state.num_non_clifford

            num_steps = 0
            game_start = time.time()
            while num_steps < max_steps:
                if state.is_terminal() or not state.has_legal_actions():
                    break
                if game_timeout > 0 and (time.time() - game_start) > game_timeout:
                    break

                action, policy, _ = mcts.select_action(state, torch.device('cpu'))
                if not policy:
                    break

                try:
                    _, done = state.apply_action(action)
                except (ValueError, KeyError, IndexError, AssertionError):
                    break

                num_steps += 1
                if done:
                    break

            final_t_gates = state.num_non_clifford
            results.append(EvalGameResult(
                initial_t_gates=initial_t_gates,
                final_t_gates=final_t_gates,
                t_gates_reduced=initial_t_gates - final_t_gates,
                num_steps=num_steps,
                simplified=state.is_terminal(),
                pyzx_t_gates=pyzx_t_gates,
                pyzx_t_gates_reduced=pyzx_reduced,
            ))

        except Exception as e:
            logging.getLogger(__name__).warning(
                f"Eval worker (seed={worker_seed}) game failed: {e}"
            )

    return results


class ParallelEvaluator(Evaluator):
    """Multi-process evaluator that distributes eval games across workers.

    Drop-in replacement for Evaluator with the same evaluate() interface.
    Internally spawns worker processes that each play a share of the eval
    games, then collects and aggregates results in the main process.

    Falls back to sequential Evaluator.evaluate() if num_workers <= 1.
    """

    def __init__(
        self,
        mcts_config: MCTSConfig,
        device: torch.device = torch.device('cpu'),
        compare_pyzx: bool = True,
        eval_temperature: float = 0.1,
        max_eval_steps: Optional[int] = None,
        game_timeout: float = 0.0,
        num_workers: int = 4,
    ):
        super().__init__(
            mcts_config=mcts_config,
            device=device,
            compare_pyzx=compare_pyzx,
            eval_temperature=eval_temperature,
            max_eval_steps=max_eval_steps,
            game_timeout=game_timeout,
        )
        self.num_workers = num_workers
        self._model_hparams: Optional[dict] = None
        self._executor: Optional[ProcessPoolExecutor] = None

        if num_workers > 1:
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
            self._executor = ProcessPoolExecutor(
                max_workers=num_workers,
                mp_context=mp.get_context('spawn'),
            )

    def evaluate(
        self,
        model: nn.Module,
        num_games: int = 20,
        num_qubits: Optional[int] = None,
        depth: Optional[int] = None,
        fixed_circuits: Optional[list] = None,
    ) -> EvalSummary:
        """Run evaluation games in parallel and return aggregated results.

        If num_workers <= 1 or fixed_circuits is provided, falls back to
        the sequential Evaluator.evaluate() (fixed_circuits contain
        unpicklable PyZX graph objects).
        """
        # Fall back to sequential for single-worker or pre-generated circuits
        if self.num_workers <= 1 or fixed_circuits is not None:
            return super().evaluate(
                model, num_games, num_qubits, depth, fixed_circuits,
            )

        start_time = time.time()

        # Extract model hparams lazily (once)
        if self._model_hparams is None:
            from alphazx.mcts.parallel_self_play import _extract_model_hparams
            self._model_hparams = _extract_model_hparams(model)

        # Serialize model weights
        state_dict = {k: v.cpu() for k, v in model.state_dict().items()}

        nq = num_qubits or self.base_config.num_qubits
        d = depth or self.base_config.depth

        # Partition games across workers
        games_per_worker = self._partition_games(num_games)

        # Per-worker seeds
        base_seed = int(time.time() * 1000) % (2**31)
        worker_seeds = [base_seed + i * 10_000 for i in range(self.num_workers)]

        # Dispatch
        logger.info(
            f"Dispatching {num_games} eval games across {self.num_workers} "
            f"workers ({games_per_worker})..."
        )
        futures: list[Future] = []
        for i, n_games in enumerate(games_per_worker):
            if n_games == 0:
                continue
            future = self._executor.submit(
                _eval_worker_play_games,
                model_state_dict=state_dict,
                model_hparams=self._model_hparams,
                mcts_config=self.base_config,
                num_games=n_games,
                worker_seed=worker_seeds[i],
                eval_temperature=self.eval_temperature,
                compare_pyzx=self.compare_pyzx,
                max_eval_steps=self.max_eval_steps,
                game_timeout=self.game_timeout,
                num_qubits=nq,
                depth=d,
            )
            futures.append(future)

        # Collect results
        logger.info(f"All {len(futures)} eval workers dispatched. Waiting...")
        all_results: list[EvalGameResult] = []
        for idx, future in enumerate(futures):
            try:
                worker_results = future.result()
                all_results.extend(worker_results)
                logger.info(
                    f"  Eval worker {idx + 1}/{len(futures)} finished: "
                    f"{len(worker_results)} games"
                )
            except Exception as e:
                logger.error(
                    f"Eval worker {idx + 1}/{len(futures)} failed: {e}. "
                    f"Skipping its games."
                )

        wall_time = time.time() - start_time
        logger.info(
            f"Parallel eval: {len(all_results)} games in {wall_time:.1f}s "
            f"across {self.num_workers} workers"
        )
        return self._aggregate_results(all_results, wall_time)

    def _partition_games(self, num_games: int) -> list[int]:
        """Partition num_games roughly equally across workers."""
        base = num_games // self.num_workers
        remainder = num_games % self.num_workers
        return [
            base + (1 if i < remainder else 0)
            for i in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        """Shutdown the process pool."""
        if self._executor is not None:
            self._executor.shutdown(wait=True)

    def __del__(self):
        try:
            if self._executor is not None:
                self._executor.shutdown(wait=False)
        except Exception:
            pass
