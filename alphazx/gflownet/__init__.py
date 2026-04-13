"""
GFlowNet training paradigm for ZX-calculus simplification.

GFlowNets (Generative Flow Networks) learn to sample trajectories
(sequences of graph rewrites) proportional to a reward function.
Unlike MCTS/AlphaZero which optimises a single best trajectory,
GFlowNets learn a *distribution* over high-reward trajectories,
which naturally handles:

- The "worsen before improving" problem (F-Right temporarily increases
  T-gates to enable future F-Left merges): GFlowNets evaluate complete
  trajectories, so intermediate worsening doesn't penalise exploration.

- Diverse solution discovery: the flow-matching objective encourages
  sampling many distinct high-reward trajectories, not just the single best.

This package reuses the shared ZX infrastructure:
- alphazx.shared.game_state.GameState for state representation
- alphazx.shared.config.CircuitConfig for circuit/episode parameters
- alphazx.shared.evaluate for model inference utilities
- alphazx.diagram / alphazx.rewriting for ZX-calculus operations
- alphazx.models for GNN backbones (GPS, HGT)
"""
