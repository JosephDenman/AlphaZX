"""
MCTS tree nodes for Sampled MCTS.

In standard AlphaZero, expansion creates a child for every legal action.
In Sampled MCTS, children are added progressively by sampling from the policy.
This avoids enumerating the combinatorial F-right action space.

Each MCTSNode stores:
- The GameState at that point in the search tree
- Visit statistics (N, W, Q) for PUCT-based selection
- A prior P from the policy network (set when the node is first created as a child)
- A dict of children keyed by action tuple
- The cached policy distribution and value from neural network evaluation
"""

from __future__ import annotations

import math
from typing import Optional

import torch

from alphazx.mcts.game_state import GameState
from alphazx.mcts.config import MCTSConfig


class MCTSNode:
    """A node in the MCTS search tree."""

    __slots__ = [
        'state', 'parent', 'action_from_parent', 'prior',
        'visit_count', 'total_value', 'reward',
        'children', 'is_expanded', 'is_terminal',
        '_cached_distribution', '_cached_value',
        '_original_prior',
    ]

    def __init__(
        self,
        state: GameState,
        parent: Optional[MCTSNode] = None,
        action_from_parent: Optional[tuple] = None,
        prior: float = 1.0,
        reward: float = 0.0,
    ):
        self.state = state
        self.parent = parent
        self.action_from_parent = action_from_parent
        self.prior = prior
        self.reward = reward  # immediate reward from parent → this node

        # Search statistics
        self.visit_count: int = 0
        self.total_value: float = 0.0

        # Children: action_tuple → MCTSNode
        # Actions are stored as tuples for hashability
        self.children: dict[tuple, MCTSNode] = {}

        # Set to True once the neural network has evaluated this node
        self.is_expanded: bool = False

        # Terminal states cannot be expanded
        self.is_terminal: bool = state.is_terminal() or not state.has_legal_actions()

        # Cached neural network outputs (set during expansion)
        self._cached_distribution = None  # AlphaZXDistribution
        self._cached_value: Optional[float] = None

        # Original prior before Dirichlet noise (set by search._apply_root_noise_if_needed)
        self._original_prior: Optional[float] = None

    @property
    def q_value(self) -> float:
        """Mean value Q = W / N. Returns 0 if unvisited."""
        if self.visit_count == 0:
            return 0.0
        return self.total_value / self.visit_count

    def puct_score(self, config: MCTSConfig) -> float:
        """
        Compute the PUCT score for selecting this node from its parent.

        Score = Q(s,a) + c_puct * P(s,a) * sqrt(N(parent)) / (1 + N(s,a))

        Higher score → more likely to be selected during tree traversal.
        The prior P biases exploration toward actions the policy network favors.
        The sqrt(N_parent)/(1+N) term ensures unvisited children are explored.
        """
        if self.parent is None:
            return 0.0
        exploration = (
            config.c_puct
            * self.prior
            * math.sqrt(self.parent.visit_count)
            / (1 + self.visit_count)
        )
        return self.q_value + exploration

    def should_widen(self, config: MCTSConfig) -> bool:
        """
        Check if this node should add a new child via progressive widening.

        Widening condition: N >= pw_c * |children|^(1/pw_alpha)

        This means: as the node accumulates visits, it earns the right to
        explore new actions. Early visits go to existing children; later
        visits trigger sampling of new children from the policy.
        """
        if self.is_terminal:
            return False
        if not self.is_expanded:
            return False
        num_children = len(self.children)
        if num_children == 0:
            return True  # Always widen if no children yet
        threshold = config.pw_c * (num_children ** (1.0 / config.pw_alpha))
        return self.visit_count >= threshold

    def select_child(self, config: MCTSConfig) -> MCTSNode:
        """Select the child with the highest PUCT score."""
        best_score = -float('inf')
        best_child = None
        for child in self.children.values():
            score = child.puct_score(config)
            if score > best_score:
                best_score = score
                best_child = child
        return best_child

    def backpropagate(self, value: float, gamma: float = 1.0) -> None:
        """
        Backpropagate a value estimate from a leaf up to the root.

        The value is discounted by gamma at each step moving upward,
        and the immediate reward at each node is added.

        For AlphaZero with gamma=1.0 on episodic tasks, this simplifies
        to propagating the raw value estimate unchanged.
        """
        node = self
        while node is not None:
            node.visit_count += 1
            node.total_value += value
            # When moving to parent, the value seen from the parent's perspective
            # includes the immediate reward of transitioning to this node
            value = node.reward + gamma * value
            node = node.parent

    @property
    def visit_count_distribution(self) -> dict[tuple, int]:
        """Return {action: visit_count} for all children."""
        return {action: child.visit_count for action, child in self.children.items()}

    def __repr__(self) -> str:
        return (
            f"MCTSNode(N={self.visit_count}, Q={self.q_value:.3f}, "
            f"P={self.prior:.3f}, children={len(self.children)}, "
            f"terminal={self.is_terminal})"
        )
