"""
TensorBoard logging for AlphaZX training.

Provides structured metric logging across the full AlphaZero loop:
- Per-step training losses and diagnostics
- Per-iteration self-play statistics
- Per-node-type graph composition over time
- Action distribution evolution
- Evaluation results vs PyZX baseline
- Replay buffer statistics
- Model weight/gradient histograms

Usage:
    logger = TBLogger("runs/alphazx_experiment_1")
    # ... in training loop ...
    logger.log_train_step(step, metrics)
    logger.log_iteration(iteration, self_play_stats, train_stats)
    logger.close()

All logging is optional — if TensorBoard is not installed, the logger
degrades gracefully to a no-op.
"""

from __future__ import annotations

import logging
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

try:
    from torch.utils.tensorboard import SummaryWriter
    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    # Use print() rather than logger here because logging may not be configured
    # at module import time — this ensures the user sees the message.
    print("[WARNING] tensorboard is not installed. TensorBoard logging disabled. "
          "Install with: pip install tensorboard")


@dataclass
class SelfPlayStats:
    """Extended statistics from a batch of self-play games.

    Collected by the Trainer from EpisodeResult objects and passed
    to the TBLogger for structured logging.
    """
    num_games: int = 0
    total_examples: int = 0
    avg_steps: float = 0.0
    avg_t_gates_reduced: float = 0.0
    simplification_rate: float = 0.0
    avg_initial_t_gates: float = 0.0
    avg_final_t_gates: float = 0.0
    avg_reward: float = 0.0
    games_with_t_increase: int = 0
    self_play_time: float = 0.0

    # Per-node-type counts (averaged over games)
    # Keys are match node type abbreviations: frz, frx, flz, flx, br, bl, yrz, yrx, ylz, ylx
    initial_node_type_counts: dict[str, float] = field(default_factory=dict)
    final_node_type_counts: dict[str, float] = field(default_factory=dict)

    # Action type distribution across all steps in this batch
    # Keys are action type names from ACTION_TYPE_NAMES
    action_type_counts: Counter = field(default_factory=Counter)


@dataclass
class TrainStepDiagnostics:
    """Extended diagnostics from a single training step.

    Beyond the basic loss values, tracks gradient norms, value prediction
    statistics, and MCTS policy entropy for monitoring training health.
    """
    policy_loss: float = 0.0
    value_loss: float = 0.0
    total_loss: float = 0.0
    grad_norm: float = 0.0
    learning_rate: float = 0.0
    value_target_mean: float = 0.0
    value_target_std: float = 0.0
    value_prediction_mean: float = 0.0
    mcts_policy_entropy: float = 0.0
    num_valid_examples: int = 0


class TBLogger:
    """TensorBoard logger for AlphaZX training.

    Wraps SummaryWriter with domain-specific logging methods.
    All methods are no-ops if TensorBoard is not installed.
    """

    def __init__(self, log_dir: str = "runs/alphazx", enabled: bool = True):
        self.enabled = enabled and _TB_AVAILABLE
        self.log_dir = log_dir

        if self.enabled:
            self._writer = SummaryWriter(log_dir=log_dir)
            logger.info(f"TensorBoard logging to {log_dir}")
        else:
            self._writer = None
            if enabled and not _TB_AVAILABLE:
                msg = ("TensorBoard logging requested but tensorboard is not installed. "
                       "Install with: pip install tensorboard")
                logger.warning(msg)
                print(f"[WARNING] {msg}")

    # ── Per-step training metrics ─────────────────────────────────────

    def log_train_step(self, global_step: int, diag: TrainStepDiagnostics) -> None:
        """Log metrics from a single training step."""
        if not self.enabled:
            return
        w = self._writer

        # Core losses
        w.add_scalar("train/policy_loss", diag.policy_loss, global_step)
        w.add_scalar("train/value_loss", diag.value_loss, global_step)
        w.add_scalar("train/total_loss", diag.total_loss, global_step)

        # Training diagnostics
        w.add_scalar("train/grad_norm", diag.grad_norm, global_step)
        w.add_scalar("train/learning_rate", diag.learning_rate, global_step)

        # Value head diagnostics
        w.add_scalar("train/value_target_mean", diag.value_target_mean, global_step)
        w.add_scalar("train/value_target_std", diag.value_target_std, global_step)
        w.add_scalar("train/value_prediction_mean", diag.value_prediction_mean, global_step)

        # Policy diagnostics
        w.add_scalar("train/mcts_policy_entropy", diag.mcts_policy_entropy, global_step)

        # Batch diagnostics
        w.add_scalar("train/valid_examples_in_batch", diag.num_valid_examples, global_step)

    # ── Per-iteration self-play metrics ───────────────────────────────

    def log_self_play(self, iteration: int, stats: SelfPlayStats) -> None:
        """Log self-play statistics for a training iteration."""
        if not self.enabled:
            return
        w = self._writer

        # Episode-level statistics
        w.add_scalar("self_play/avg_steps", stats.avg_steps, iteration)
        w.add_scalar("self_play/avg_t_gates_reduced", stats.avg_t_gates_reduced, iteration)
        w.add_scalar("self_play/simplification_rate", stats.simplification_rate, iteration)
        w.add_scalar("self_play/avg_initial_t_gates", stats.avg_initial_t_gates, iteration)
        w.add_scalar("self_play/avg_final_t_gates", stats.avg_final_t_gates, iteration)
        w.add_scalar("self_play/avg_reward", stats.avg_reward, iteration)
        w.add_scalar("self_play/games_with_t_increase", stats.games_with_t_increase, iteration)
        w.add_scalar("self_play/num_examples", stats.total_examples, iteration)
        w.add_scalar("self_play/wall_time_seconds", stats.self_play_time, iteration)

        # Per-node-type counts — initial composition
        for ntype, count in stats.initial_node_type_counts.items():
            w.add_scalar(f"node_types_initial/{ntype}", count, iteration)

        # Per-node-type counts — final composition
        for ntype, count in stats.final_node_type_counts.items():
            w.add_scalar(f"node_types_final/{ntype}", count, iteration)

        # Action type distribution
        total_actions = sum(stats.action_type_counts.values())
        if total_actions > 0:
            for action_name, count in stats.action_type_counts.items():
                fraction = count / total_actions
                w.add_scalar(f"actions/{action_name}_fraction", fraction, iteration)
                w.add_scalar(f"actions/{action_name}_count", count, iteration)

    # ── Per-iteration training aggregates ─────────────────────────────

    def log_iteration_training(
        self,
        iteration: int,
        avg_policy_loss: float,
        avg_value_loss: float,
        avg_total_loss: float,
        training_time: float,
        buffer_size: int,
        buffer_total_added: int,
    ) -> None:
        """Log iteration-level training aggregates and buffer statistics."""
        if not self.enabled:
            return
        w = self._writer

        w.add_scalar("iteration/avg_policy_loss", avg_policy_loss, iteration)
        w.add_scalar("iteration/avg_value_loss", avg_value_loss, iteration)
        w.add_scalar("iteration/avg_total_loss", avg_total_loss, iteration)
        w.add_scalar("iteration/training_time_seconds", training_time, iteration)

        w.add_scalar("buffer/size", buffer_size, iteration)
        w.add_scalar("buffer/total_added", buffer_total_added, iteration)

    # ── Evaluation metrics ────────────────────────────────────────────

    def log_evaluation(self, iteration: int, eval_summary) -> None:
        """Log evaluation results (EvalSummary from evaluator.py)."""
        if not self.enabled:
            return
        w = self._writer

        w.add_scalar("eval/avg_t_gates_reduced", eval_summary.avg_t_gates_reduced, iteration)
        w.add_scalar("eval/avg_reduction_ratio", eval_summary.avg_reduction_ratio, iteration)
        w.add_scalar("eval/simplification_rate", eval_summary.simplification_rate, iteration)
        w.add_scalar("eval/avg_steps", eval_summary.avg_steps, iteration)
        w.add_scalar("eval/wall_time_seconds", eval_summary.wall_time, iteration)

        # PyZX comparison
        if eval_summary.pyzx_avg_t_gates_reduced is not None:
            w.add_scalar("eval/pyzx_avg_t_gates_reduced",
                         eval_summary.pyzx_avg_t_gates_reduced, iteration)
            w.add_scalar("eval/pyzx_avg_reduction_ratio",
                         eval_summary.pyzx_avg_reduction_ratio, iteration)
            total_games = (eval_summary.agent_vs_pyzx_wins
                           + eval_summary.agent_vs_pyzx_ties
                           + eval_summary.agent_vs_pyzx_losses)
            if total_games > 0:
                w.add_scalar("eval/vs_pyzx_win_rate",
                             eval_summary.agent_vs_pyzx_wins / total_games, iteration)

    # ── Model weight/gradient histograms ──────────────────────────────

    def log_model_histograms(self, iteration: int, model: nn.Module) -> None:
        """Log weight and gradient histograms for all model parameters.

        Call once per iteration (not per step) to avoid excessive overhead.
        """
        if not self.enabled:
            return
        w = self._writer

        for name, param in model.named_parameters():
            if param.requires_grad:
                w.add_histogram(f"weights/{name}", param.data, iteration)
                if param.grad is not None:
                    w.add_histogram(f"gradients/{name}", param.grad, iteration)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def flush(self) -> None:
        """Flush pending events to disk."""
        if self.enabled and self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        """Close the TensorBoard writer."""
        if self.enabled and self._writer is not None:
            self._writer.close()
            logger.info("TensorBoard writer closed")
