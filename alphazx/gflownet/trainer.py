"""
GFlowNet training loop for ZX-calculus simplification.

Implements Trajectory Balance (TB) loss:

    L_TB = (log Z + Σ_t log P_F(a_t|s_t) - log R(x) - Σ_t log P_B(a_t|s_{t+1}))^2

The training loop:
1. Sample a batch of trajectories from the forward policy P_F.
2. Compute TB loss for each trajectory.
3. Average and backpropagate.
4. Periodically evaluate T-gate reduction on held-out circuits.
"""

from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.optim as optim

import pyzx

from alphazx.gflownet.config import GFlowNetConfig
from alphazx.gflownet.environment import ZXGFlowNetEnv
from alphazx.gflownet.policy import GFlowNetForwardPolicy, UniformBackwardPolicy
from alphazx.gflownet.replay_buffer import PrioritizedReplayBuffer, ReplayEntry
from alphazx.gflownet.sampler import (
    TrajectorySampler, AnnotatedTrajectory, ParallelTrajectorySampler,
)
from alphazx.shared.constants import ACTION_TYPE_NAMES

logger = logging.getLogger(__name__)


@dataclass
class GFlowNetTrainMetrics:
    """Diagnostics from a single training step."""
    tb_loss: float = 0.0
    log_Z: float = 0.0
    mean_sub_steps: float = 0.0       # mean sub-action transitions per trajectory
    mean_rewrites: float = 0.0         # mean actual ZX diagram rewrites per trajectory
    mean_terminal_reward: float = 0.0
    mean_shaped_reward: float = 0.0    # reward after intermediate shaping
    mean_t_gate_reduction: float = 0.0
    max_t_gate_reduction: int = 0
    frac_positive_reduction: float = 0.0
    learning_rate: float = 0.0
    grad_norm: float = 0.0
    num_trajectories: int = 0
    num_replay_trajectories: int = 0   # how many came from replay buffer
    replay_buffer_size: int = 0
    replay_buffer_unique_fps: int = 0  # unique action-type fingerprints
    reward_exponent: float = 0.0       # current annealed exponent
    step_time: float = 0.0


@dataclass
class GFlowNetEvalMetrics:
    """Diagnostics from an evaluation round."""
    mean_t_gate_reduction: float = 0.0
    mean_reduction_ratio: float = 0.0
    max_t_gate_reduction: int = 0
    frac_positive_reduction: float = 0.0
    num_games: int = 0
    eval_time: float = 0.0
    # PyZX baseline comparison
    pyzx_mean_t_gate_reduction: Optional[float] = None
    pyzx_mean_reduction_ratio: Optional[float] = None
    agent_wins: int = 0
    agent_ties: int = 0
    agent_losses: int = 0


class GFlowNetTrainer:
    """Trajectory Balance training loop for ZX-calculus GFlowNet.

    Usage::

        config = GFlowNetConfig(num_qubits=5, depth=5)
        model = AlphaZXModel(...)  # same model as MCTS
        trainer = GFlowNetTrainer(model, config, device='cpu')

        for iteration in range(100):
            metrics = trainer.train_step()
            print(f"iter={iteration}  TB_loss={metrics.tb_loss:.4f}  "
                  f"log_Z={metrics.log_Z:.3f}  "
                  f"T_reduced={metrics.mean_t_gate_reduction:.2f}")

            if iteration % 10 == 0:
                eval_metrics = trainer.evaluate()
                print(f"  eval: T_reduced={eval_metrics.mean_t_gate_reduction:.2f}  "
                      f"positive={eval_metrics.frac_positive_reduction:.1%}")
    """

    def __init__(
        self,
        model: nn.Module,
        config: GFlowNetConfig,
        device: str = 'cpu',
    ):
        self.config = config
        self.device = device

        # Forward policy wraps the model
        self.policy = GFlowNetForwardPolicy(model, pe_dim=config.pe_dim)
        self.policy.to(device)

        # Environment
        self.env = ZXGFlowNetEnv(config)

        # Whether replay is active
        self._replay_active = config.replay_ratio > 0

        # Trajectory sampler
        self.sampler = TrajectorySampler(
            env=self.env,
            policy=self.policy,
            device=device,
            temperature=config.sampling_temperature,
            epsilon_uniform=config.epsilon_uniform,
            max_trajectory_length=config.max_trajectory_length,
            reward_exponent=config.reward_exponent,
            min_reward=config.min_reward,
            reward_shaping_coeff=config.reward_shaping_coeff,
            retain_states_for_replay=self._replay_active,
        )

        # Optimizer — separate LR for log_Z (single scalar) vs model params.
        # log_Z gradient is swamped by the ~585K model parameters if they
        # share the same LR; a 10-100x higher LR for log_Z is standard
        # practice in TB-loss GFlowNets (Malkin et al., 2022).
        model_params = list(self.policy.model.parameters())
        self.optimizer = optim.Adam([
            {"params": model_params,
             "lr": config.learning_rate,
             "weight_decay": config.weight_decay},
            {"params": [self.policy.log_Z],
             "lr": config.log_z_learning_rate,
             "weight_decay": 0.0},
        ])

        # LR scheduler
        self.scheduler: Optional[optim.lr_scheduler._LRScheduler] = None
        if config.lr_schedule == 'cosine':
            # Decays to eta_min once over T_max steps.
            # Caller can set T_max after knowing total iterations.
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=1000, eta_min=1e-5,
            )
        elif config.lr_schedule == 'cosine_restarts':
            # Cosine annealing with warm restarts every T_0 iterations.
            # Prevents LR from decaying to zero permanently, allowing
            # continued learning throughout training.
            self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                self.optimizer,
                T_0=config.lr_restart_period,
                T_mult=1,       # keep period constant across restarts
                eta_min=1e-5,
            )

        # Parallel trajectory sampler (optional)
        self._parallel_sampler: Optional[ParallelTrajectorySampler] = None
        if config.num_sampling_workers > 0:
            self._parallel_sampler = ParallelTrajectorySampler(
                model=model,
                config=config,
                sampler=self.sampler,
                num_workers=config.num_sampling_workers,
            )

        # Prioritized replay buffer
        self.replay_buffer = PrioritizedReplayBuffer(
            max_size=config.replay_buffer_size,
            min_reward=config.replay_min_reward,
            diversity_weight=config.replay_diversity_weight,
            min_reduction_ratio=config.replay_min_reduction_ratio,
        )
        self._step_count = 0

    # ------------------------------------------------------------------
    # Reward exponent annealing
    # ------------------------------------------------------------------

    def _current_reward_exponent(self) -> float:
        """Compute the annealed reward exponent for the current step.

        Linearly interpolates from ``reward_exponent_initial`` to
        ``reward_exponent`` over the warmup period.
        """
        cfg = self.config
        if cfg.reward_exponent_warmup_iters <= 0:
            return cfg.reward_exponent
        progress = min(1.0, self._step_count / cfg.reward_exponent_warmup_iters)
        return (cfg.reward_exponent_initial
                + (cfg.reward_exponent - cfg.reward_exponent_initial) * progress)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_step(self) -> GFlowNetTrainMetrics:
        """Run one training step: sample trajectories, compute TB loss, update.

        When ``replay_ratio > 0``, a fraction of each batch comes from the
        prioritized replay buffer.  Replay trajectories are re-evaluated
        through the current policy to produce proper gradients (teacher-forced
        rollouts).  After each step, high-reward fresh trajectories are added
        to the buffer.

        When ``reward_shaping_coeff > 0``, the shaped reward (which includes
        intermediate T-gate reduction bonuses) is used in the TB loss instead
        of the raw terminal reward.
        """
        t0 = time.time()
        cfg = self.config
        self.policy.train()

        # --- Apply reward exponent annealing ---
        current_exponent = self._current_reward_exponent()
        self.sampler.reward_exponent = current_exponent

        # --- Determine batch composition ---
        n_replay = 0
        n_fresh = cfg.trajectories_per_batch
        if self._replay_active and len(self.replay_buffer) > 0:
            n_replay = max(1, int(cfg.trajectories_per_batch * cfg.replay_ratio))
            n_fresh = cfg.trajectories_per_batch - n_replay

        # --- Sample fresh trajectories ---
        if self._parallel_sampler is not None:
            # Parallel: workers sample without grad, main process replays
            # with grad.  This is the sample-then-replay architecture.
            trajectories = self._parallel_sampler.sample_and_replay(
                n_fresh, reward_exponent=current_exponent,
            )
        else:
            # Sequential: sample with gradients in the main process
            trajectories = self.sampler.sample_batch_with_grad(n_fresh)

        # --- Add qualifying fresh trajectories to replay buffer ---
        if self._replay_active:
            for traj in trajectories:
                if not traj.rewrite_actions:
                    continue
                # Two replay paths:
                # 1. Sequential sampling → initial_state is a live GameState
                # 2. Parallel sampling → _source_record has pre-exported tensors
                #    (avoids hash-dependent match-diagram reconstruction)
                has_state = traj.initial_state is not None
                has_record = traj._source_record is not None
                if not has_state and not has_record:
                    continue
                entry = ReplayEntry(
                    initial_state=traj.initial_state,  # may be None for parallel path
                    action_tuples=list(traj.rewrite_actions),
                    terminal_reward=traj.terminal_reward,
                    shaped_reward=traj.shaped_reward,
                    initial_t_gates=traj.initial_t_gates,
                    final_t_gates=traj.final_t_gates,
                    per_rewrite_t_deltas=list(traj.per_rewrite_t_deltas),
                    action_type_fingerprint=traj.action_type_fingerprint,
                    trajectory_record=traj._source_record,
                )
                self.replay_buffer.add(entry)

        # --- Replay trajectories (re-evaluated through current policy) ---
        replay_trajectories: list[AnnotatedTrajectory] = []
        if n_replay > 0:
            replay_entries = self.replay_buffer.sample(n_replay)
            # Partition entries by replay strategy
            state_entries = [e for e in replay_entries if e.initial_state is not None]
            record_entries = [e for e in replay_entries if e.initial_state is None
                             and e.trajectory_record is not None]
            # State-based replay (sequential sampling path)
            for entry in state_entries:
                replay_traj = self.sampler.replay_rollout(
                    entry.initial_state, entry.action_tuples,
                )
                replay_trajectories.append(replay_traj)
            # Tensor-based replay (parallel sampling path)
            if record_entries:
                from alphazx.gflownet.sampler import TrajectoryRecord
                records = [e.trajectory_record for e in record_entries]
                replayed = self.sampler.replay_records_batched(records)
                replay_trajectories.extend(replayed)

        all_trajectories = trajectories + replay_trajectories

        # --- Compute loss (TB or SubTB depending on config) ---
        use_subtb = cfg.loss_type == 'sub_trajectory_balance'
        tb_losses = []
        for traj in all_trajectories:
            if len(traj) == 0:
                continue
            if use_subtb:
                loss = self._sub_trajectory_balance_loss(traj)
            else:
                loss = self._trajectory_balance_loss(traj)
            if loss is not None:
                tb_losses.append(loss)

        if not tb_losses:
            return GFlowNetTrainMetrics()

        total_loss = torch.stack(tb_losses).mean()

        # Backward + optimize
        self.optimizer.zero_grad()
        total_loss.backward()

        # Gradient norm (for diagnostics)
        grad_norm = 0.0
        for p in self.policy.parameters():
            if p.grad is not None:
                grad_norm += p.grad.data.norm(2).item() ** 2
        grad_norm = grad_norm ** 0.5

        # Clip gradients (configurable max_norm)
        torch.nn.utils.clip_grad_norm_(
            self.policy.parameters(), max_norm=cfg.grad_clip_max_norm,
        )

        self.optimizer.step()
        if self.scheduler is not None:
            self.scheduler.step()

        self._step_count += 1

        # Collect metrics (from fresh trajectories only for consistency)
        t_reductions = [t.t_gate_reduction for t in trajectories]
        rewards = [t.terminal_reward for t in trajectories]
        shaped_rewards = [t.shaped_reward for t in trajectories]
        sub_steps = [len(t) for t in trajectories]
        rewrites = [t.num_rewrites for t in trajectories]

        current_lr = self.optimizer.param_groups[0]['lr']  # model LR
        n = max(1, len(trajectories))

        return GFlowNetTrainMetrics(
            tb_loss=total_loss.item(),
            log_Z=self.policy.log_Z.item(),
            mean_sub_steps=sum(sub_steps) / n,
            mean_rewrites=sum(rewrites) / n,
            mean_terminal_reward=sum(rewards) / n,
            mean_shaped_reward=sum(shaped_rewards) / n,
            mean_t_gate_reduction=sum(t_reductions) / n,
            max_t_gate_reduction=max(t_reductions) if t_reductions else 0,
            frac_positive_reduction=sum(1 for r in t_reductions if r > 0) / n,
            learning_rate=current_lr,
            grad_norm=grad_norm,
            num_trajectories=len(trajectories),
            num_replay_trajectories=len(replay_trajectories),
            replay_buffer_size=len(self.replay_buffer),
            replay_buffer_unique_fps=self.replay_buffer.num_unique_fingerprints,
            reward_exponent=current_exponent,
            step_time=time.time() - t0,
        )

    def _trajectory_balance_loss(
        self, traj: AnnotatedTrajectory,
    ) -> Optional[torch.Tensor]:
        """Compute TB loss for a single trajectory.

        L_TB = (log Z + Σ log P_F(a_t|s_t) - log R(x) - Σ log P_B(a_t|s_{t+1}))^2

        When reward shaping is active (``reward_shaping_coeff > 0``), the
        shaped reward is used in place of the raw terminal reward.  The
        shaped reward includes multiplicative bonuses for intermediate
        T-gate reductions, providing a richer training signal.

        log Z is a learnable parameter (self.policy.log_Z).
        Σ log P_F is differentiable through the model.
        log R(x) is a constant (terminal reward; shaped_reward is deprecated).
        Σ log P_B is a constant (uniform backward policy).
        """
        if len(traj) == 0 or traj.terminal_reward <= 0:
            return None

        # Use shaped reward if active (deprecated), otherwise terminal reward
        effective_reward = traj.shaped_reward if traj.shaped_reward > 0 else traj.terminal_reward

        log_Z = self.policy.log_Z
        sum_log_pf = traj.sum_log_pf        # differentiable
        log_R = math.log(effective_reward)   # constant
        sum_log_pb = traj.sum_log_pb         # constant (uniform P_B)

        # TB condition: log Z + Σ log P_F = log R + Σ log P_B
        # Loss = (log Z + Σ log P_F - log R - Σ log P_B)^2
        delta = log_Z + sum_log_pf - log_R - sum_log_pb
        return delta ** 2

    def _sub_trajectory_balance_loss(
        self, traj: AnnotatedTrajectory,
    ) -> Optional[torch.Tensor]:
        """Compute SubTB(λ) loss for a single trajectory (vectorized).

        SubTB (Madan et al., 2023) sums flow-matching constraints over
        all contiguous sub-trajectories, weighted by λ^(length-1):

            L_SubTB = Σ_{0≤i<j≤n} λ^(j-i-1) ×
                      (log F(s_i) + Σ log P_F - log F(s_j) - Σ log P_B)²

        where n = number of rewrites, and:
          - log F(s_0) = log Z              (learnable partition function)
          - log F(s_n) = log R(x)           (terminal reward)
          - log F(s_k) for 0<k<n = model value head output (state flow)

        Sub-trajectories are defined at the REWRITE level (not sub-step
        level), since intermediate flow estimates only exist at rewrite
        boundaries.  The log P_F and log P_B for a rewrite window (i,j)
        are the sums over all sub-steps in rewrites i through j-1.

        **Vectorized implementation:**  The per-pair delta simplifies to

            delta[i,j] = h[i] - h[j]

        where ``h[k] = flows[k] - cum_pf[k] + cum_pb[k]``.  This allows
        the entire O(n²) sum to be computed as a single outer-subtraction
        on a vector of length n+1, drastically reducing the number of
        autograd graph nodes and enabling BLAS-level parallelism.
        """
        n = traj.num_rewrites
        if n == 0 or traj.terminal_reward <= 0:
            return None

        lam = self.config.subtb_lambda
        effective_reward = (traj.shaped_reward
                            if traj.shaped_reward > 0
                            else traj.terminal_reward)
        log_R = math.log(effective_reward)

        # --- Per-rewrite log_pf (differentiable) and log_pb (scalar) ---
        per_rewrite_log_pf = []
        per_rewrite_log_pb = []

        for k in range(n):
            start = traj.rewrite_start_indices[k]
            end = (traj.rewrite_start_indices[k + 1]
                   if k + 1 < n
                   else len(traj.transitions))
            sub_steps = traj.transitions[start:end]
            if not sub_steps:
                per_rewrite_log_pf.append(torch.tensor(0.0))
                per_rewrite_log_pb.append(0.0)
                continue
            log_pf_sum = torch.stack(
                [t.log_pf.squeeze() for t in sub_steps],
            ).sum()
            log_pb_sum = sum(t.log_pb for t in sub_steps)
            per_rewrite_log_pf.append(log_pf_sum)
            per_rewrite_log_pb.append(log_pb_sum)

        # --- Prefix sums as tensors ---
        # cum_pf[k] = Σ_{r=0}^{k-1} per_rewrite_log_pf[r]   (differentiable)
        # cum_pb[k] = Σ_{r=0}^{k-1} per_rewrite_log_pb[r]   (constant)
        pf_stack = torch.stack(per_rewrite_log_pf)          # [n]
        cum_pf = torch.cat([
            torch.zeros(1), torch.cumsum(pf_stack, dim=0),
        ])                                                   # [n+1]

        pb_arr = torch.tensor(per_rewrite_log_pb)            # [n]
        cum_pb = torch.cat([
            torch.zeros(1), torch.cumsum(pb_arr, dim=0),
        ])                                                   # [n+1]

        # --- Flow estimates at rewrite boundaries ---
        log_Z = self.policy.log_Z
        flows_list: list = [log_Z]
        for k in range(1, n):
            if k < len(traj.rewrite_boundary_log_flows):
                flows_list.append(traj.rewrite_boundary_log_flows[k])
            else:
                flows_list.append(log_Z)
        flows_list.append(torch.tensor(log_R))
        flows = torch.stack(flows_list)                      # [n+1]

        # --- Vectorized SubTB loss ---
        # Key identity:
        #   delta[i,j] = flows[i] + (cum_pf[j]-cum_pf[i])
        #              - flows[j] - (cum_pb[j]-cum_pb[i])
        #              = (flows[i] - cum_pf[i] + cum_pb[i])
        #              - (flows[j] - cum_pf[j] + cum_pb[j])
        #              = h[i] - h[j]
        h = flows - cum_pf + cum_pb                          # [n+1]

        # Pairwise deltas via outer subtraction
        delta = h.unsqueeze(1) - h.unsqueeze(0)              # [n+1, n+1]

        # Weight matrix: w[i,j] = λ^(j-i-1) for j > i, 0 otherwise
        idx = torch.arange(n + 1)
        spans = idx.unsqueeze(0) - idx.unsqueeze(1)          # [n+1, n+1]  (j - i in transposed form)
        # We want j > i, so spans = j - i > 0 in row-i, col-j layout
        # spans[i,j] = j - i when using (i=row, j=col)
        spans_ij = idx.unsqueeze(1) - idx.unsqueeze(0)       # spans_ij[i,j] = j - i
        mask = spans_ij > 0
        weights = torch.where(
            mask,
            lam ** (spans_ij.float() - 1),
            torch.zeros(1),
        )                                                     # [n+1, n+1]

        weighted_sq = weights * delta ** 2
        subtb_loss = weighted_sq.sum()
        weight_sum = weights.sum()

        if weight_sum > 0:
            subtb_loss = subtb_loss / weight_sum

        return subtb_loss

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @staticmethod
    def _run_pyzx_baseline(pyzx_graph) -> tuple[Optional[int], Optional[int]]:
        """Run PyZX's full_reduce on the SAME circuit the agent evaluated.

        Returns (final_t_gates, t_gates_reduced) or (None, None) on failure.
        """
        if pyzx_graph is None:
            return None, None
        try:
            initial_t = pyzx.tcount(pyzx_graph)
            reduced = pyzx_graph.copy()
            pyzx.full_reduce(reduced)
            final_t = pyzx.tcount(reduced)
            return final_t, initial_t - final_t
        except Exception as e:
            logger.warning(f"PyZX baseline failed: {e}")
            return None, None

    @torch.no_grad()
    def evaluate(self, num_games: int = 10) -> GFlowNetEvalMetrics:
        """Evaluate current policy on random circuits.

        Samples trajectories at the configured eval temperature and measures
        T-gate reduction.  Also runs PyZX's ``full_reduce`` on the same
        circuits for an apples-to-apples baseline comparison.
        """
        t0 = time.time()
        self.policy.eval()

        # Create sampler for evaluation (with PyZX graph retention)
        eval_sampler = TrajectorySampler(
            env=self.env,
            policy=self.policy,
            device=self.device,
            temperature=self.config.eval_temperature,
            epsilon_uniform=0.0,
            max_trajectory_length=self.config.max_trajectory_length,
            reward_exponent=self.config.reward_exponent,
            min_reward=self.config.min_reward,
        )

        trajectories = eval_sampler.sample_batch_with_pyzx(num_games)

        # --- Agent metrics ---
        t_reductions = [t.t_gate_reduction for t in trajectories]
        reduction_ratios = []
        for t in trajectories:
            if t.initial_t_gates > 0:
                reduction_ratios.append(t.t_gate_reduction / t.initial_t_gates)
            else:
                reduction_ratios.append(0.0)

        n = max(1, len(trajectories))

        # --- PyZX baseline ---
        pyzx_reductions = []
        pyzx_ratios = []
        wins, ties, losses = 0, 0, 0

        for t in trajectories:
            pyzx_final, pyzx_reduced = self._run_pyzx_baseline(t.pyzx_graph)
            if pyzx_reduced is not None:
                pyzx_reductions.append(pyzx_reduced)
                if t.initial_t_gates > 0:
                    pyzx_ratios.append(pyzx_reduced / t.initial_t_gates)
                else:
                    pyzx_ratios.append(0.0)
                # Compare agent vs PyZX
                agent_red = t.t_gate_reduction
                if agent_red > pyzx_reduced:
                    wins += 1
                elif agent_red == pyzx_reduced:
                    ties += 1
                else:
                    losses += 1

        pyzx_mean_red = (sum(pyzx_reductions) / len(pyzx_reductions)
                         if pyzx_reductions else None)
        pyzx_mean_ratio = (sum(pyzx_ratios) / len(pyzx_ratios)
                           if pyzx_ratios else None)

        return GFlowNetEvalMetrics(
            mean_t_gate_reduction=sum(t_reductions) / n,
            mean_reduction_ratio=sum(reduction_ratios) / n,
            max_t_gate_reduction=max(t_reductions) if t_reductions else 0,
            frac_positive_reduction=sum(1 for r in t_reductions if r > 0) / n,
            num_games=len(trajectories),
            eval_time=time.time() - t0,
            pyzx_mean_t_gate_reduction=pyzx_mean_red,
            pyzx_mean_reduction_ratio=pyzx_mean_ratio,
            agent_wins=wins,
            agent_ties=ties,
            agent_losses=losses,
        )

    @torch.no_grad()
    def evaluate_benchmarks(
        self,
        benchmarks: list,
    ) -> list[dict]:
        """Evaluate the agent on fixed benchmark circuits.

        Runs the agent on each benchmark circuit (deterministic starting
        state) and compares against PyZX's ``full_reduce`` on the same
        circuit.

        Parameters
        ----------
        benchmarks : list[BenchmarkCircuit]
            Loaded benchmark circuits (from ``alphazx.diagram.benchmark_circuits``).

        Returns
        -------
        list[dict]
            Per-circuit results with keys: name, family, qubits, initial_t,
            agent_final_t, agent_reduced, pyzx_final_t, pyzx_reduced, winner.
        """
        from alphazx.shared.game_state import GameState

        self.policy.eval()
        eval_sampler = TrajectorySampler(
            env=self.env,
            policy=self.policy,
            device=self.device,
            temperature=self.config.eval_temperature,
            epsilon_uniform=0.0,
            max_trajectory_length=self.config.max_trajectory_length,
            reward_exponent=self.config.reward_exponent,
            min_reward=self.config.min_reward,
        )

        results = []
        for bc in benchmarks:
            # Run agent on the benchmark circuit
            state = GameState.from_diagram(bc.zx_diagram.copy())
            try:
                traj = eval_sampler.rollout_from_state(state, bc.pyzx_graph)
            except ValueError as e:
                # Circuit has phases outside the model's vocabulary.
                # Log and skip rather than crashing the whole eval.
                logger.warning("Skipping benchmark %s: %s", bc.name, e)
                results.append({
                    "name": bc.name,
                    "family": bc.family,
                    "qubits": bc.qubits,
                    "initial_t": bc.t_count,
                    "agent_final_t": None,
                    "agent_reduced": None,
                    "pyzx_final_t": None,
                    "pyzx_reduced": None,
                    "winner": "skipped",
                    "error": str(e),
                })
                continue

            agent_reduced = traj.t_gate_reduction

            # Run PyZX baseline on the same circuit
            pyzx_final, pyzx_reduced = self._run_pyzx_baseline(bc.pyzx_graph)
            if pyzx_reduced is None:
                pyzx_reduced = 0
                pyzx_final = bc.t_count

            # Determine winner
            if agent_reduced > pyzx_reduced:
                winner = "agent"
            elif agent_reduced == pyzx_reduced:
                winner = "tie"
            else:
                winner = "pyzx"

            results.append({
                "name": bc.name,
                "family": bc.family,
                "qubits": bc.qubits,
                "initial_t": bc.t_count,
                "agent_final_t": traj.final_t_gates,
                "agent_reduced": agent_reduced,
                "pyzx_final_t": pyzx_final,
                "pyzx_reduced": pyzx_reduced,
                "winner": winner,
            })

        return results
