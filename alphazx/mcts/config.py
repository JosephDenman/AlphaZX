"""
Configuration for Sampled MCTS.

Unlike standard AlphaZero MCTS which enumerates all legal actions at expansion time,
Sampled MCTS progressively widens the tree by sampling actions from the policy network.
This avoids the combinatorial explosion of the F-right action space (16 phases × 5 edges × 2^degree).
"""

from dataclasses import dataclass


@dataclass
class MCTSConfig:
    """Configuration for Sampled MCTS search."""

    # --- Search budget ---
    num_simulations: int = 100
    """Number of MCTS simulations per move."""

    # --- PUCT exploration ---
    c_puct: float = 1.5
    """Exploration constant in the PUCT formula:
    Q(a) + c_puct * P(a) * sqrt(N_parent) / (1 + N(a))"""

    # --- Progressive widening ---
    pw_alpha: float = 0.5
    """Progressive widening exponent. A node is widened (new child sampled) when:
    N(node) >= pw_c * num_children(node) ^ (1 / pw_alpha)
    Lower alpha → more aggressive widening (more children explored).
    0.5 is a standard starting point from the literature."""

    pw_c: float = 1.0
    """Progressive widening coefficient. Controls the base rate of widening.
    Higher values delay widening (require more visits before adding children)."""

    # --- Root exploration noise ---
    dirichlet_alpha: float = 0.3
    """Dirichlet noise concentration parameter for root node exploration.
    Smaller values produce spikier noise (more aggressive exploration).
    0.3 is standard for Go-sized action spaces; may need tuning for ZX."""

    dirichlet_epsilon: float = 0.25
    """Fraction of root prior that is replaced with Dirichlet noise.
    P_root(a) = (1 - epsilon) * P_network(a) + epsilon * Dir(alpha)"""

    # --- Temperature ---
    temperature: float = 1.0
    """Temperature for action selection from root visit counts.
    pi(a) = N(a)^(1/tau) / sum(N(a')^(1/tau))
    1.0 for training (proportional to visits), 0.1 for evaluation (near-greedy)."""

    # --- Value discount ---
    gamma: float = 1.0
    """Discount factor for backpropagated values. 1.0 for episodic tasks
    (standard in AlphaZero). < 1.0 if you want to prefer shorter solutions."""

    # --- Positional encoding ---
    pe_dim: int = 20
    """Dimension of random-walk positional encoding used in preprocessing."""

    # --- Virtual loss (for future parallel MCTS) ---
    virtual_loss: float = 0.0
    """Virtual loss added during selection to discourage threads from
    exploring the same path. 0.0 disables (single-threaded search)."""

    # --- Game parameters (used when creating fresh games for self-play) ---
    num_qubits: int = 5
    depth: int = 5
    max_episode_length: int = 100
    step_penalty: float = 1.0
    simplified_reward: float = 1000.0

    # --- Episode termination ---
    max_t_gate_increase: int = 5
    """Terminate episode early if T-gates exceed initial count by this amount.
    Prevents runaway degeneration where the agent spends 100 steps making
    the diagram progressively worse. Set to 0 to disable."""

    min_initial_t_gates: int = 2
    """Minimum T-gates required to start an episode. Circuits with fewer
    T-gates provide negligible learning signal and waste compute. The
    generator will re-roll until it finds a circuit meeting this threshold."""

    max_circuit_retries: int = 20
    """Maximum attempts to generate a circuit meeting min_initial_t_gates
    before falling back to whatever was generated."""

    # --- Circuit generation ---
    circuit_type: str = 'cnot_had_phase'
    """Which circuit generator to use for self-play:
    - 'cnot_had_phase': Generate from CNOT+Hadamard+Phase circuits (recommended).
      These are real quantum circuits converted to ZX diagrams, with realistic
      structure and plenty of T-gates available for optimization.
    - 'clifford': Generate random ZX graphs directly via pyzx.generate.cliffords.
      These are random graph-level ZX diagrams, not derived from circuits.
      May have fewer T-gates and less realistic structure for optimization."""

    p_had: float = 0.2
    """Probability of Hadamard gates in CNOT_HAD_PHASE circuits."""

    p_t: float = 0.2
    """Probability of T-gates (vs other phase gates) in CNOT_HAD_PHASE circuits."""
