"""
Sampled MCTS search.

This implements AlphaZero-style MCTS with progressive widening, adapted
for the structured action space of ZX-calculus diagram simplification.

Key differences from standard AlphaZero MCTS:
1. Actions are SAMPLED from the policy network, not enumerated.
2. Children are added progressively as a node accumulates visits.
3. Duplicate samples (same action sampled twice) are detected and collapsed.
4. The policy prior for PUCT is computed from the cached distribution via log_prob.

The search loop:
    for each simulation:
        1. SELECT: walk down tree using PUCT until reaching a leaf or widening point
        2. EXPAND: if widening, sample a new action and create a child; OR
                   if leaf, evaluate with neural network
        3. BACKPROPAGATE: propagate the value estimate back up to root
"""

from __future__ import annotations

import logging
import math
import time
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from alphazx.mcts.config import MCTSConfig
from alphazx.mcts.game_state import GameState
from alphazx.mcts.node import MCTSNode
from alphazx.mcts.evaluate import evaluate_state, compute_action_prior

logger = logging.getLogger(__name__)


class MCTS:
    """Sampled Monte Carlo Tree Search with progressive widening."""

    def __init__(self, model: nn.Module, config: MCTSConfig):
        self.model = model
        self.config = config
        self.model.eval()

    def search(
        self,
        root_state: GameState,
        device: torch.device = torch.device('cpu'),
    ) -> dict[tuple, float]:
        """
        Run MCTS from the given root state and return a policy target.

        :param root_state: The current game state to search from
        :param device: Device for neural network inference
        :return: Dictionary mapping action tuples to visit-count-based probabilities.
                 This is the MCTS policy target used for training.
        """
        root = MCTSNode(state=root_state)

        # Evaluate and expand the root node
        self._expand_node(root, device)

        # Track how many root children have received Dirichlet noise so far.
        self._root_noise_applied_count = 0
        self._apply_root_noise_if_needed(root)

        # Profiling accumulators
        _t_expand = 0.0
        _t_clone = 0.0
        _n_expand = 0
        _n_clone = 0

        # Run simulations
        for _ in range(self.config.num_simulations):
            node = root
            search_path = [node]

            # === SELECT ===
            while node.is_expanded and not node.is_terminal:
                if node.should_widen(self.config):
                    # Progressive widening: sample a new child
                    _t0 = time.time()
                    new_child = self._try_add_child(node)
                    _t_clone += time.time() - _t0
                    _n_clone += 1
                    if new_child is not None:
                        if node is root:
                            self._apply_root_noise_if_needed(root)
                        search_path.append(new_child)
                        node = new_child
                        break  # Evaluate the new leaf

                if not node.children:
                    break

                # Select best existing child
                node = node.select_child(self.config)
                search_path.append(node)

            # === EXPAND & EVALUATE ===
            value = 0.0
            if node.is_terminal:
                value = 0.0
            elif not node.is_expanded:
                _t0 = time.time()
                value = self._expand_node(node, device)
                _t_expand += time.time() - _t0
                _n_expand += 1
            else:
                value = node._cached_value if node._cached_value is not None else 0.0

            # === BACKPROPAGATE ===
            node.backpropagate(value, self.config.gamma)

        logger.debug(
            f"MCTS search: {self.config.num_simulations} sims, "
            f"expand={_n_expand} ({_t_expand:.3f}s), "
            f"clone/add_child={_n_clone} ({_t_clone:.3f}s), "
            f"children={len(root.children)}"
        )

        # Return the visit-count policy from root
        return self._compute_policy(root)

    def _expand_node(self, node: MCTSNode, device: torch.device) -> float:
        """
        Evaluate a node with the neural network and mark it as expanded.

        Returns the value estimate for backpropagation.
        """
        if node.is_terminal:
            node.is_expanded = True
            node._cached_value = 0.0
            return 0.0

        distribution, value = evaluate_state(
            self.model, node.state, self.config.pe_dim, device
        )

        node._cached_distribution = distribution
        node._cached_value = value
        node.is_expanded = True

        # Sample an initial set of children to get the tree started.
        # Without at least one child, PUCT selection has nothing to work with.
        self._try_add_child(node)

        return value

    def _try_add_child(self, node: MCTSNode, max_attempts: int = 10) -> Optional[MCTSNode]:
        """
        Sample an action from the node's cached distribution and add it as a child.

        If the sampled action already exists as a child, retry up to max_attempts times.
        Returns the new child node, or None if all attempts produced duplicates.
        """
        if node._cached_distribution is None:
            return None

        for _ in range(max_attempts):
            # Sample one action from the policy
            action_tensor = node._cached_distribution.sample(1)  # 1 x B x L
            action_tuple = tuple(action_tensor.squeeze().tolist())

            if action_tuple in node.children:
                continue  # Duplicate — try again

            # Compute prior probability for PUCT
            prior = compute_action_prior(node._cached_distribution, action_tuple)

            # Create the child state by cloning and applying the action
            try:
                child_state = node.state.clone()
                reward, done = child_state.apply_action(action_tuple)
            except (ValueError, KeyError, IndexError, AssertionError):
                # Invalid action (can happen with sampling from learned policy)
                # Skip and try another sample
                continue

            child = MCTSNode(
                state=child_state,
                parent=node,
                action_from_parent=action_tuple,
                prior=max(prior, 1e-8),  # Ensure non-zero prior
                reward=reward,
            )
            node.children[action_tuple] = child
            return child

        return None  # All attempts produced duplicates or invalid actions

    def _apply_root_noise_if_needed(self, root: MCTSNode) -> None:
        """
        Re-apply Dirichlet noise to ALL root children if new children were added.

        Standard AlphaZero applies Dirichlet noise to the full root prior
        at once. With progressive widening, children are added incrementally.
        To ensure every root child gets exploration noise, we re-generate
        and re-apply noise whenever the child count changes.

        Each child's prior is: (1 - ε) * network_prior + ε * noise[i]

        To re-apply correctly, we store the original network prior
        (child._original_prior) so that re-noising doesn't compound.
        """
        num_children = len(root.children)
        if num_children == 0 or num_children == self._root_noise_applied_count:
            return  # No new children since last noise application

        epsilon = self.config.dirichlet_epsilon
        alpha = self.config.dirichlet_alpha
        noise = np.random.dirichlet([alpha] * num_children)

        for i, child in enumerate(root.children.values()):
            # Save original prior on first noise application
            if child._original_prior is None:
                child._original_prior = child.prior
            child.prior = (1 - epsilon) * child._original_prior + epsilon * noise[i]

        self._root_noise_applied_count = num_children

    def _compute_policy(self, root: MCTSNode) -> dict[tuple, float]:
        """
        Convert root visit counts into a probability distribution (the MCTS policy target).

        With temperature τ:
            π(a) = N(a)^(1/τ) / Σ_a' N(a')^(1/τ)

        τ = 1.0: proportional to visits (more exploration, used for training)
        τ → 0:   greedy (pick the most visited action, used for evaluation)
        """
        if not root.children:
            return {}

        tau = self.config.temperature
        visit_counts = {
            action: child.visit_count
            for action, child in root.children.items()
        }

        if tau < 0.01:
            # Near-greedy: put all mass on the most-visited action
            best_action = max(visit_counts, key=visit_counts.get)
            return {action: (1.0 if action == best_action else 0.0)
                    for action in visit_counts}

        # Apply temperature with numerical stability (log-space softmax)

        log_probs = {
            action: (1.0 / tau) * math.log(max(count, 1))
            for action, count in visit_counts.items()
        }
        max_log_prob = max(log_probs.values())
        exp_probs = {
            action: math.exp(lp - max_log_prob)
            for action, lp in log_probs.items()
        }
        total = sum(exp_probs.values())

        return {action: prob / total for action, prob in exp_probs.items()}

    def select_action(
        self,
        root_state: GameState,
        device: torch.device = torch.device('cpu'),
    ) -> tuple[tuple, dict[tuple, float], float]:
        """
        Convenience method: run search and return the selected action.

        :return: (selected_action, mcts_policy, root_value) where:
            - selected_action is sampled from the MCTS policy
            - mcts_policy is the full visit-count distribution (training target)
            - root_value is the neural network's value estimate for the root state
        """
        policy = self.search(root_state, device)

        if not policy:
            # No legal actions — return a dummy
            return (), {}, 0.0

        # Sample an action from the MCTS policy
        actions = list(policy.keys())
        probs = list(policy.values())
        idx = np.random.choice(len(actions), p=probs)
        selected_action = actions[idx]

        # Get root value (the neural network's estimate before search)
        # The search may have updated this via backpropagation, but we return
        # the raw network estimate as the value target is the episode outcome
        root_value = 0.0  # Will be filled in by self-play with actual outcome

        return selected_action, policy, root_value
