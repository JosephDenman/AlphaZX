"""
Sampled MCTS search with batched leaf evaluation.

This implements AlphaZero-style MCTS with progressive widening, adapted
for the structured action space of ZX-calculus diagram simplification.

Key differences from standard AlphaZero MCTS:
1. Actions are SAMPLED from the policy network, not enumerated.
2. Children are added progressively as a node accumulates visits.
3. Duplicate samples (same action sampled twice) are detected and collapsed.
4. The policy prior for PUCT is computed from the cached distribution via log_prob.
5. Leaf nodes are evaluated in batches for efficiency (virtual loss prevents
   path duplication within each wave).

The search loop (batched):
    while simulations remain:
        WAVE (batch_size simulations):
            for each simulation:
                1. SELECT: walk down tree using PUCT until leaf or widen point
                2. If leaf needs NN eval: collect it, apply virtual loss to path
                3. If terminal or already expanded: note value for backprop
            BATCH EVALUATE all collected leaves at once
            REMOVE virtual loss from all paths
            BACKPROPAGATE all simulations
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
from alphazx.mcts.evaluate import evaluate_state, evaluate_states_batch, compute_action_prior

logger = logging.getLogger(__name__)


class MCTS:
    """Sampled Monte Carlo Tree Search with progressive widening and batched evaluation."""

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

        Simulations are run in waves of `config.leaf_batch_size`. Within each
        wave, virtual loss is applied to encourage path diversity, and all
        unexpanded leaves are batch-evaluated in a single neural network
        forward pass.

        :param root_state: The current game state to search from
        :param device: Device for neural network inference
        :return: Dictionary mapping action tuples to visit-count-based probabilities.
                 This is the MCTS policy target used for training.
        """
        root = MCTSNode(state=root_state)

        # Evaluate and expand the root node (always single-state)
        self._expand_node(root, device)

        # Track how many root children have received Dirichlet noise so far.
        self._root_noise_applied_count = 0
        self._apply_root_noise_if_needed(root)

        # Profiling accumulators
        _t_expand = 0.0
        _t_clone = 0.0
        _n_expand = 0
        _n_clone = 0

        leaf_batch_size = self.config.leaf_batch_size
        sims_done = 0

        while sims_done < self.config.num_simulations:
            wave_size = min(leaf_batch_size, self.config.num_simulations - sims_done)

            # ================================================================
            # Phase 1: Selection — run wave_size simulations, collecting leaves
            # ================================================================
            wave = []  # list of (leaf_node, search_path, needs_nn_eval)
            unique_eval_nodes = {}  # id(node) -> node (dedup within wave)

            for _ in range(wave_size):
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
                            break  # New child is unexpanded — will be evaluated

                    if not node.children:
                        break

                    # Select best existing child
                    node = node.select_child(self.config)
                    search_path.append(node)

                # Classify this simulation's leaf
                needs_eval = not node.is_expanded and not node.is_terminal
                if needs_eval:
                    unique_eval_nodes[id(node)] = node

                wave.append((node, search_path, needs_eval))

                # Virtual loss: increment visit counts on path nodes to
                # discourage subsequent simulations in this wave from
                # selecting the same path.
                if needs_eval:
                    for n in search_path:
                        n.visit_count += 1

                sims_done += 1

            # ================================================================
            # Phase 2: Batch evaluate all unique unexpanded leaves
            # ================================================================
            if unique_eval_nodes:
                eval_nodes = list(unique_eval_nodes.values())
                _t0 = time.time()
                states = [n.state for n in eval_nodes]
                results = evaluate_states_batch(
                    self.model, states, self.config.pe_dim, device
                )
                _t_expand += time.time() - _t0
                _n_expand += len(eval_nodes)

                for eval_node, (dist, value) in zip(eval_nodes, results):
                    eval_node._cached_distribution = dist
                    eval_node._cached_value = value
                    eval_node.is_expanded = True
                    # Seed diverse children so MCTS explores all action types
                    _t0 = time.time()
                    seeded = self._seed_diverse_children(eval_node)
                    if seeded == 0:
                        self._try_add_child(eval_node)
                    _t_clone += time.time() - _t0
                    _n_clone += max(seeded, 1)

            # ================================================================
            # Phase 3: Remove virtual loss, then real backpropagation
            # ================================================================
            # First, undo all virtual losses
            for leaf, search_path, needs_eval in wave:
                if needs_eval:
                    for n in search_path:
                        n.visit_count -= 1

            # Then, backpropagate each simulation with real values
            for leaf, search_path, needs_eval in wave:
                if leaf.is_terminal:
                    value = 0.0
                elif leaf._cached_value is not None:
                    value = leaf._cached_value
                else:
                    value = 0.0
                leaf.backpropagate(value, self.config.gamma)

            # Re-apply Dirichlet noise if root children changed during this wave
            self._apply_root_noise_if_needed(root)

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

        Used only for the root node (which is always evaluated individually).
        All other nodes are batch-evaluated in the search loop.

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

        # Seed one child per available action type so MCTS explores the
        # full action-type space from the start, not just the majority
        # type (F-Right).  Falls back to a single sample if seeding fails.
        seeded = self._seed_diverse_children(node)
        if seeded == 0:
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

    def _seed_diverse_children(self, node: MCTSNode) -> int:
        """Seed a node with one child per available action type.

        In ZX-calculus simplification the action distribution is heavily
        skewed toward F-Right (which has far more match nodes than F-Left
        or B-rules).  Standard progressive widening samples children
        proportional to the policy, so the first ~5 children are almost
        always F-Right variants.  This prevents MCTS from ever evaluating
        F-Left or B-Right subtrees when the simulation budget is small.

        This method creates one child for each action type that has
        non-zero probability in the distribution, ensuring MCTS gets a
        diverse initial set to explore.  Subsequent widening still samples
        proportionally, so the long-tail bias is only corrected at the
        start.

        Returns the number of children successfully added.
        """
        if node._cached_distribution is None:
            return 0

        dist = node._cached_distribution
        mixture_probs = dist.mixture_dist_params  # (B, T)
        num_types = mixture_probs.shape[-1]
        added = 0

        # Track which action types already have a child
        existing_types = set()
        for action_tuple in node.children:
            if len(action_tuple) > 1:
                existing_types.add(int(action_tuple[1]))

        for action_type in range(num_types):
            if action_type in existing_types:
                continue
            # Skip action types with negligible probability
            prob = mixture_probs[0, action_type].item() if mixture_probs.dim() > 1 else mixture_probs[action_type].item()
            if prob < 1e-6:
                continue

            # Sample an action of this specific type
            import torch as _torch
            for _attempt in range(5):
                action_type_tensor = _torch.tensor([[action_type]])  # (1, B)
                try:
                    nodes = dist.sample_nodes(action_type_tensor)[0]
                    phases = dist.sample_phases(action_type_tensor, nodes)[0]
                    new_edges = dist.sample_new_edges(action_type_tensor, nodes)[0]
                    transfer_edges = dist.sample_transfer_edges(action_type_tensor, nodes)[0]

                    # Build action tuple: (graph_id, action_type, node, phase, new_edge, *transfer)
                    gid = dist.graph_ids[0].item() if dist.graph_ids is not None else 0
                    action_tuple = (
                        gid,
                        action_type,
                        nodes.squeeze().item(),
                        phases.squeeze().item(),
                        new_edges.squeeze().item(),
                        *transfer_edges.squeeze().tolist(),
                    )
                    action_tuple = tuple(int(x) for x in action_tuple)

                    if action_tuple in node.children:
                        continue

                    prior = compute_action_prior(dist, action_tuple)
                    child_state = node.state.clone()
                    reward, done = child_state.apply_action(action_tuple)

                    child = MCTSNode(
                        state=child_state,
                        parent=node,
                        action_from_parent=action_tuple,
                        prior=max(prior, 1e-8),
                        reward=reward,
                    )
                    node.children[action_tuple] = child
                    added += 1
                    break
                except (ValueError, KeyError, IndexError, AssertionError):
                    continue

        return added

    def _apply_root_noise_ext(self, root: MCTSNode, noise_applied_count: int) -> int:
        """Apply Dirichlet noise to root children (external count version).

        Like :meth:`_apply_root_noise_if_needed`, but takes and returns the
        noise-applied count as a parameter instead of using instance state.
        Used by :meth:`search_batch` which manages K independent searches.
        """
        num_children = len(root.children)
        if num_children == 0 or num_children == noise_applied_count:
            return noise_applied_count

        epsilon = self.config.dirichlet_epsilon
        alpha = self.config.dirichlet_alpha
        noise = np.random.dirichlet([alpha] * num_children)

        for i, child in enumerate(root.children.values()):
            if child._original_prior is None:
                child._original_prior = child.prior
            child.prior = (1 - epsilon) * child._original_prior + epsilon * noise[i]

        return num_children

    def _apply_root_noise_if_needed(self, root: MCTSNode) -> None:
        """
        Re-apply Dirichlet noise to ALL root children if new children were added.

        Standard AlphaZero applies Dirichlet noise to the full root prior
        at once. With progressive widening, children are added incrementally.
        To ensure every root child gets exploration noise, we re-generate
        and re-apply noise whenever the child count changes.

        Each child's prior is: (1 - epsilon) * network_prior + epsilon * noise[i]

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

    # ------------------------------------------------------------------
    # Cross-game batched search
    # ------------------------------------------------------------------

    def search_batch(
        self,
        root_states: list[GameState],
        device: torch.device = torch.device('cpu'),
    ) -> list[dict[tuple, float]]:
        """Run MCTS search for multiple game states simultaneously.

        This is the cross-game batched variant of :meth:`search`.  Instead
        of evaluating leaf nodes from a single search tree, it collects
        leaves across *all* K active trees each wave and evaluates them in
        a single combined forward pass.  This increases batch utilisation
        and typically gives 2–3× throughput on CPU for small graphs.

        The per-tree search logic (selection, progressive widening, virtual
        loss, backpropagation) is identical to :meth:`search`.

        :param root_states: List of K GameStates to search from.
        :param device: Device for neural network inference.
        :return: List of K policy dicts (same format as :meth:`search`).
        """
        K = len(root_states)
        if K == 0:
            return []
        if K == 1:
            return [self.search(root_states[0], device)]

        # --- Initialise K root nodes ---
        roots: list[MCTSNode] = [MCTSNode(state=s) for s in root_states]
        noise_counts = [0] * K

        # Batch-expand non-terminal roots in one forward pass
        non_terminal = [(i, r) for i, r in enumerate(roots) if not r.is_terminal]
        for r in roots:
            if r.is_terminal:
                r.is_expanded = True
                r._cached_value = 0.0

        if non_terminal:
            nt_states = [r.state for _, r in non_terminal]
            nt_results = evaluate_states_batch(
                self.model, nt_states, self.config.pe_dim, device,
            )
            for (_, root), (dist, value) in zip(non_terminal, nt_results):
                root._cached_distribution = dist
                root._cached_value = value
                root.is_expanded = True
                seeded = self._seed_diverse_children(root)
                if seeded == 0:
                    self._try_add_child(root)

        for g in range(K):
            noise_counts[g] = self._apply_root_noise_ext(roots[g], noise_counts[g])

        # --- Per-game search state ---
        sims_done = [0] * K
        leaf_batch_size = self.config.leaf_batch_size
        num_sims = self.config.num_simulations

        # --- Main search loop (waves in lockstep across all games) ---
        while any(sd < num_sims for sd in sims_done):

            # Phase 1: Selection across all active games
            all_eval_nodes: list[MCTSNode] = []
            # (game_idx, wave, eval_offset, eval_count)
            game_waves: list[tuple[int, list, int, int]] = []

            for g in range(K):
                if sims_done[g] >= num_sims:
                    continue

                root = roots[g]
                wave_size = min(leaf_batch_size, num_sims - sims_done[g])
                wave: list[tuple] = []
                unique_eval_nodes: dict[int, MCTSNode] = {}

                for _ in range(wave_size):
                    node = root
                    search_path = [node]

                    # === SELECT ===
                    while node.is_expanded and not node.is_terminal:
                        if node.should_widen(self.config):
                            new_child = self._try_add_child(node)
                            if new_child is not None:
                                if node is root:
                                    noise_counts[g] = self._apply_root_noise_ext(
                                        root, noise_counts[g],
                                    )
                                search_path.append(new_child)
                                node = new_child
                                break

                        if not node.children:
                            break

                        node = node.select_child(self.config)
                        search_path.append(node)

                    needs_eval = not node.is_expanded and not node.is_terminal
                    if needs_eval:
                        unique_eval_nodes[id(node)] = node

                    wave.append((node, search_path, needs_eval))

                    # Virtual loss
                    if needs_eval:
                        for n in search_path:
                            n.visit_count += 1

                    sims_done[g] += 1

                eval_nodes_list = list(unique_eval_nodes.values())
                eval_offset = len(all_eval_nodes)
                all_eval_nodes.extend(eval_nodes_list)
                game_waves.append((g, wave, eval_offset, len(eval_nodes_list)))

            # Phase 2: Combined batch evaluation across all games
            if all_eval_nodes:
                eval_states = [n.state for n in all_eval_nodes]
                all_results = evaluate_states_batch(
                    self.model, eval_states, self.config.pe_dim, device,
                )
            else:
                all_results = []

            # Phase 3: Expansion, virtual-loss removal, and backpropagation
            for g, wave, eval_offset, eval_count in game_waves:
                # Apply NN results to evaluated nodes
                for j in range(eval_count):
                    eval_node = all_eval_nodes[eval_offset + j]
                    dist, value = all_results[eval_offset + j]
                    eval_node._cached_distribution = dist
                    eval_node._cached_value = value
                    eval_node.is_expanded = True
                    seeded = self._seed_diverse_children(eval_node)
                    if seeded == 0:
                        self._try_add_child(eval_node)

                # Remove virtual loss
                for leaf, search_path, needs_eval in wave:
                    if needs_eval:
                        for n in search_path:
                            n.visit_count -= 1

                # Backpropagate
                for leaf, search_path, needs_eval in wave:
                    if leaf.is_terminal:
                        value = 0.0
                    elif leaf._cached_value is not None:
                        value = leaf._cached_value
                    else:
                        value = 0.0
                    leaf.backpropagate(value, self.config.gamma)

                # Re-apply root noise if new children were added
                noise_counts[g] = self._apply_root_noise_ext(
                    roots[g], noise_counts[g],
                )

        return [self._compute_policy(root) for root in roots]

    def _compute_policy(self, root: MCTSNode) -> dict[tuple, float]:
        """
        Convert root visit counts into a probability distribution (the MCTS policy target).

        With temperature tau:
            pi(a) = N(a)^(1/tau) / sum_a' N(a')^(1/tau)

        tau = 1.0: proportional to visits (more exploration, used for training)
        tau -> 0:  greedy (pick the most visited action, used for evaluation)
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
