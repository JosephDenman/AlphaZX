"""
Shared configuration for ZX-calculus circuit generation, episode control, and rewards.

These parameters are used by all training paradigms (MCTS/AlphaZero, GFlowNet, etc.).
Paradigm-specific parameters live in their own config modules
(e.g. alphazx.mcts.config.MCTSConfig, alphazx.gflownet.config.GFlowNetConfig).
"""

from dataclasses import dataclass


@dataclass
class CircuitConfig:
    """Configuration for circuit generation, episode control, and reward shaping.

    These parameters are paradigm-agnostic: they describe the *environment*
    (what circuits look like, when episodes end, how rewards are computed),
    not the *search algorithm* (MCTS simulations, PUCT constants, etc.).
    """

    # --- Circuit generation ---
    num_qubits: int = 5
    depth: int = 5
    circuit_type: str = 'cnot_had_phase'
    """Which circuit generator to use:
    - 'cnot_had_phase': CNOT+Hadamard+Phase circuits (recommended for training).
    - 'clifford': Random ZX graphs via pyzx.generate.cliffords.
    - 'cliffordT': Clifford+T circuits via pyzx.generate.cliffordT.
      Uses Riu et al. (2025) default gate probabilities (p_t, p_s, p_hsh).
      This is the benchmark distribution for comparison with their PPO paper."""

    p_had: float = 0.2
    """Probability of Hadamard gates in CNOT_HAD_PHASE circuits."""

    p_t: float = 0.2
    """Probability of T-gates (vs other phase gates) in CNOT_HAD_PHASE circuits.
    For cliffordT circuits, this is the T-gate probability (Riu et al. default: 0.17)."""

    p_s: float = 0.24
    """Probability of S-gates in cliffordT circuits (Riu et al. default: 0.24).
    Only used when circuit_type='cliffordT'."""

    p_hsh: float = 0.25
    """Probability of HSH (Hadamard sandwich) gates in cliffordT circuits
    (Riu et al. default: 0.25).  Only used when circuit_type='cliffordT'."""

    min_initial_t_gates: int = 2
    """Minimum T-gates required to start an episode."""

    max_circuit_retries: int = 20
    """Maximum attempts to generate a circuit meeting min_initial_t_gates."""

    # --- Episode control ---
    max_episode_length: int = 100

    max_t_gate_increase: int = 5
    """Terminate episode early if T-gates exceed initial count by this amount."""

    adaptive_episode_length: bool = True
    """Scale max_episode_length based on circuit complexity."""

    episode_length_factor: int = 5
    """Multiplier for adaptive episode length:
    max_steps = min(max_episode_length, factor * num_qubits * depth)."""

    # --- Reward / value ---
    step_penalty: float = 1.0
    simplified_reward: float = 1000.0

    value_target_mode: str = 'discounted_return'
    """How to compute value targets:
    - 'discounted_return': Per-step discounted returns.
    - 'uniform_outcome': All steps share the same target."""

    gamma: float = 0.99
    """Discount factor for value computation."""

    # --- Positional encoding ---
    pe_dim: int = 20
    """Dimension of random-walk positional encoding."""

    @property
    def effective_max_episode_length(self) -> int:
        """Compute the effective max episode length, accounting for adaptive scaling."""
        if self.adaptive_episode_length:
            adaptive = self.episode_length_factor * self.num_qubits * self.depth
            return min(self.max_episode_length, adaptive)
        return self.max_episode_length
