"""
Configuration for GFlowNet training.

GFlowNetConfig extends the shared CircuitConfig with GFlowNet-specific
parameters (loss objective, flow architecture, sampling strategy, etc.).
"""

from dataclasses import dataclass

from alphazx.shared.config import CircuitConfig


@dataclass
class GFlowNetConfig(CircuitConfig):
    """Configuration for GFlowNet training on ZX-calculus simplification.

    Inherits all circuit/episode/reward parameters from CircuitConfig.
    Adds GFlowNet-specific parameters below.
    """

    # --- Loss objective ---
    loss_type: str = 'trajectory_balance'
    """GFlowNet loss objective:
    - 'trajectory_balance' (TB): log Z + Σ log P_F(a|s) = log R(x) + Σ log P_B(a|s).
      Most common, works well for medium-length trajectories.
    - 'sub_trajectory_balance' (SubTB): TB applied to sub-trajectories.
      Better credit assignment for long trajectories.
    - 'detailed_balance' (DB): per-transition flow conservation.
      More stable but requires learning edge flows.
    - 'flow_matching' (FM): per-state flow conservation.
      No backward policy needed, but requires learning state flows."""

    # --- Reward ---
    reward_exponent: float = 4.0
    """Exponent applied to the reward: R(x) = (T_reduced / T_initial) ^ exponent.
    Higher exponent concentrates sampling on the best trajectories.
    Lower exponent encourages more diversity."""

    reward_exponent_initial: float = 1.0
    """Starting reward exponent for annealing.  When annealing is active
    (reward_exponent_warmup_iters > 0), the exponent linearly increases
    from this value to ``reward_exponent`` over the warmup period.
    Starting low (1.0) ensures almost all trajectories get non-trivial
    rewards early in training, avoiding the "everything gets min_reward"
    collapse.  Set equal to ``reward_exponent`` to disable annealing."""

    reward_exponent_warmup_iters: int = 200
    """Number of training iterations over which to anneal the reward
    exponent from ``reward_exponent_initial`` to ``reward_exponent``.
    0 = no annealing (use ``reward_exponent`` from the start)."""

    min_reward: float = 0.01
    """Minimum reward floor applied after exponentiation (prevents
    log(0) in TB loss). Trajectories that increase T-gate count still
    receive min_reward.  Keep this large enough that log(min_reward) is
    not much bigger than the typical sum_log_pf; 0.01 gives log ≈ -4.6."""

    # --- Intermediate reward shaping (DEPRECATED) ---
    reward_shaping_coeff: float = 0.0
    """DEPRECATED — leave at 0.0.  Coefficient for intermediate reward
    shaping via exponential bonus on per-step T-gate reductions:

        R_shaped = R_terminal × exp(coeff × Σ max(0, δT_t) / T_initial)

    This is redundant with SubTB's built-in intermediate flow estimation
    (state flow F(s_k) at rewrite boundaries), which already handles
    credit assignment over long trajectories.  Empirically, the
    exponential accumulation causes shaped_R to span 7+ orders of
    magnitude on small circuits with long trajectories, producing
    gradient explosion.  See Madan et al. (2023) on SubTB and Pan et al.
    (2023, GAFlowNets) on why naive intermediate reward shaping in
    GFlowNets leads to bias and instability.

    0.0 = no shaping (pure terminal reward, recommended)."""

    # --- Flow architecture ---
    learn_log_Z: bool = True
    """Learn the partition function log Z as a trainable parameter.
    Only used with trajectory_balance loss."""

    # --- Sampling ---
    trajectories_per_batch: int = 64
    """Number of complete trajectories to sample per training batch."""

    epsilon_uniform: float = 0.0
    """Probability of sampling actions uniformly (exploration).
    0.0 = pure policy sampling, 1.0 = pure uniform sampling."""

    max_trajectory_length: int = 20
    """Maximum number of rewrite steps per trajectory. Trajectories
    can terminate early if no improving actions are available.
    Start with shorter trajectories (10-20) for faster iteration
    and increase once the policy starts finding improvements."""

    # --- Backward policy ---
    learn_backward_policy: bool = True
    """Whether to learn a parametric backward policy P_B.
    If False, uses uniform backward policy (simpler but less expressive).
    Only relevant for TB and SubTB losses."""

    # --- Temperature ---
    sampling_temperature: float = 1.0
    """Temperature for forward policy sampling during training.
    Higher = more exploration, lower = more exploitation."""

    eval_temperature: float = 1.0
    """Temperature for evaluation.  GFlowNets are trained to sample
    trajectories ∝ R(x), so evaluation should use the same temperature
    as training (1.0) to measure whether the learned distribution
    concentrates on high-reward trajectories.  Low temperatures (e.g.
    0.1) produce degenerate greedy sequences that the TB objective was
    never optimized for, causing a severe train/eval gap."""

    # --- Training ---
    learning_rate: float = 1e-3
    log_z_learning_rate: float = 5e-2
    """Learning rate for the log Z parameter.  log Z is a single scalar
    whose gradient is swamped by the much larger model gradient.  Using a
    separate, higher LR (typically 10-100x the model LR) is standard
    practice in GFlowNet training (Malkin et al., 2022).  0.1 causes
    log_Z to overshoot monotonically; 0.01 is too slow (only reaches
    −1.5 after 300 iters); 0.05 (50x model LR) balances convergence
    speed and stability."""
    weight_decay: float = 1e-4
    lr_schedule: str = 'cosine_restarts'
    """LR schedule: 'cosine' decays to eta_min once, 'cosine_restarts'
    uses warm restarts every T_0 iterations, 'constant' keeps LR fixed."""
    lr_restart_period: int = 50
    """Period (in iterations) for cosine warm restarts (T_0)."""
    # --- SubTB(λ) ---
    subtb_lambda: float = 0.9
    """Interpolation parameter for Sub-Trajectory Balance loss (Madan et
    al., 2023).  λ=0 gives Detailed Balance (local, per-transition),
    λ=1 gives full Trajectory Balance (global).  Intermediate values
    (0.8–0.95) provide dense credit assignment over sub-trajectories of
    varying length, which helps with 40-70 sub-step trajectories.
    Only used when ``loss_type == 'sub_trajectory_balance'``."""

    # --- Gradient clipping ---
    grad_clip_max_norm: float = 1.0
    """Maximum gradient norm for clipping.  SubTB loss produces O(n^2)
    squared-delta terms whose gradients are inherently larger than TB's
    single-delta gradient, so aggressive clipping (1.0) is important
    for stability.  Values above 2.0 tend to cause instability with
    SubTB on long trajectories."""

    # --- Replay buffer ---
    replay_buffer_size: int = 1000
    """Maximum number of trajectories stored in the replay buffer.
    Older low-priority entries are evicted when full."""
    replay_ratio: float = 0.0
    """Fraction of each batch drawn from replay buffer (vs fresh samples).
    0.0 = pure on-policy (no replay).  Recommended: 0.25–0.5."""
    replay_min_reward: float = 0.02
    """(Deprecated — use ``replay_min_reduction_ratio`` instead.)
    Minimum shaped reward for replay buffer admission.  With high
    ``reward_exponent`` values this threshold becomes unreachable because
    rewards are exponentiated before comparison.  Kept for backward
    compatibility; ignored when ``replay_min_reduction_ratio > 0``."""
    replay_min_reduction_ratio: float = 0.02
    """Minimum T-count reduction ratio (fraction of initial T-gates
    eliminated) for a trajectory to be admitted to the replay buffer.
    Compared against ``(initial_T - final_T) / initial_T``, which is
    independent of the reward exponent.  0.02 = at least 2% T-count
    reduction.  Set to 0.0 to disable this filter (falls back to
    ``replay_min_reward`` on shaped_reward)."""
    replay_diversity_weight: float = 0.1
    """Weight of the diversity bonus in replay priority scoring.
    Priority = reward_score + diversity_weight × diversity_score.
    Higher values encourage replaying trajectories with rare action-type
    patterns over repeatedly replaying the single best trajectory."""

    # --- Parallel sampling ---
    num_sampling_workers: int = 0
    """Number of parallel worker processes for trajectory sampling.
    0 = sequential sampling in the main process (original behavior).
    When > 0, workers sample trajectories without gradients using frozen
    model weights, then the main process replays recorded actions through
    the current model with gradients for loss computation.
    Recommended: 2-8 depending on CPU cores available."""
