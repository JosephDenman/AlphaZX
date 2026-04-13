"""
Configuration for Sampled MCTS.

MCTSConfig extends the shared CircuitConfig with MCTS-specific parameters
(PUCT constants, progressive widening, Dirichlet noise, batching, etc.).
"""

from dataclasses import dataclass

from alphazx.shared.config import CircuitConfig


@dataclass
class MCTSConfig(CircuitConfig):
    """Configuration for Sampled MCTS search.

    Inherits all circuit/episode/reward parameters from CircuitConfig.
    Adds MCTS-specific parameters below.
    """

    # --- Search budget ---
    num_simulations: int = 800
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

    # --- Batched leaf evaluation ---
    leaf_batch_size: int = 8
    """Number of leaf nodes to collect before batch-evaluating with the neural
    network.  Batching turns N sequential forward passes into ceil(N/batch_size)
    batched passes, giving ~3-5x speedup on the NN portion of MCTS.
    Virtual loss is applied during each wave to encourage path diversity.
    Set to 1 to disable batching (sequential evaluation, original behavior)."""

    # --- Cross-game batching ---
    concurrent_games: int = 1
    """Number of games to play concurrently within each worker process.
    When > 1, MCTS searches for K games are interleaved so that leaf
    evaluations from all K trees are combined into a single forward pass.
    This increases batch utilisation and typically gives 2-3x throughput
    on CPU for small graphs.  Set to 1 to disable (sequential games)."""

    # --- Torch compilation ---
    torch_compile: bool = False
    """Enable torch.compile() for the neural network forward pass.
    Fuses operations and optimises memory-access patterns, giving
    roughly 1.5-3x inference speedup on CPU after a one-time warmup
    compilation cost (~10-30 s per worker on first forward pass).
    Requires PyTorch >= 2.0.  Disabled by default because PyG's
    heterogeneous message-passing may not compile cleanly on all
    platforms."""
