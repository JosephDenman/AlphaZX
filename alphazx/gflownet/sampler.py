"""
Trajectory sampling for GFlowNet training — decomposed action design.

Each rewrite action is decomposed into sequential sub-actions that each
become a separate transition in the GFlowNet DAG.  This aligns with the
directed graphical model of the policy distribution:

    s → type → node → [phase → new_edge → transfer_edges]  (F-Right)
    s → type → node                                          (non-F-Right)

Each sub-action gets its own (log_pf, log_pb) in the TB loss, giving the
flow equations fine-grained credit assignment over individual decisions.

Sub-action states
-----------------
A "GFlowNet state" is either:
  - A real ZX diagram (GameState)  — has a meaningful reward
  - A partial action:  (GameState, type, [node, [phase, [new_edge]]])
    — intermediate state, no reward, exists only for flow bookkeeping

The trajectory alternates between real ZX states and sequences of
partial-action sub-steps that culminate in applying a rewrite.
"""

from __future__ import annotations

import logging
import math
import multiprocessing as mp
import os
import random
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from alphazx.diagram import POSSIBLE_PHASES, NUM_POSSIBLE_NEW_EDGES
from alphazx.gflownet.environment import ZXGFlowNetEnv
from alphazx.gflownet.policy import GFlowNetForwardPolicy
from alphazx.shared.game_state import GameState

# Constants for backward policy (uniform over possible values)
_NUM_POSSIBLE_PHASES = len(POSSIBLE_PHASES)
_NUM_POSSIBLE_NEW_EDGES = NUM_POSSIBLE_NEW_EDGES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sub-action phase tracking
# ---------------------------------------------------------------------------

class SubActionPhase(Enum):
    """Which sub-decision comes next in the decomposed action."""
    CHOOSE_TYPE = auto()        # → choose action type
    CHOOSE_NODE = auto()        # → choose node index
    CHOOSE_PHASE = auto()       # → choose phase (F-Right only)
    CHOOSE_NEW_EDGE = auto()    # → choose new_edge (F-Right only)
    CHOOSE_TRANSFER = auto()    # → choose transfer_edges (F-Right only)
    APPLY = auto()              # → assemble & apply the rewrite


@dataclass
class PartialAction:
    """Tracks the partially-specified action being built up sub-step by sub-step."""
    action_type: Optional[int] = None
    node_index: Optional[int] = None
    phase: Optional[int] = None
    new_edge: Optional[int] = None
    transfer_edges: Optional[list[int]] = None

    @property
    def phase(self) -> SubActionPhase:
        """Which sub-decision comes next."""
        if self.action_type is None:
            return SubActionPhase.CHOOSE_TYPE
        if self.node_index is None:
            return SubActionPhase.CHOOSE_NODE
        # Non-F-Right (types 2-9): ready to apply after node
        if self.action_type >= 2:
            return SubActionPhase.APPLY
        # F-Right (types 0-1): still need phase, edge, transfer
        if self._phase_val is None:
            return SubActionPhase.CHOOSE_PHASE
        if self.new_edge is None:
            return SubActionPhase.CHOOSE_NEW_EDGE
        if self.transfer_edges is None:
            return SubActionPhase.CHOOSE_TRANSFER
        return SubActionPhase.APPLY

    def to_action_tuple(self) -> tuple:
        """Assemble the complete action tuple for env.step()."""
        graph_id = 0
        at = self.action_type
        ni = self.node_index
        ph = self._phase_val if self._phase_val is not None else 0
        ne = self.new_edge if self.new_edge is not None else 0
        te = self.transfer_edges if self.transfer_edges is not None else []
        return (graph_id, at, ni, ph, ne, *te)


# Fix: 'phase' is used as both a property (SubActionPhase) and a field (int).
# Rename the int field to _phase_val internally.

@dataclass
class PartialAction:
    """Tracks the partially-specified action being built up sub-step by sub-step."""
    action_type: Optional[int] = None
    node_index: Optional[int] = None
    phase_val: Optional[int] = None    # the chosen phase index
    new_edge: Optional[int] = None
    transfer_edges: Optional[list[int]] = None

    @property
    def next_phase(self) -> SubActionPhase:
        """Which sub-decision comes next."""
        if self.action_type is None:
            return SubActionPhase.CHOOSE_TYPE
        if self.node_index is None:
            return SubActionPhase.CHOOSE_NODE
        # Non-F-Right (types 2-9): ready to apply after node
        if self.action_type >= 2:
            return SubActionPhase.APPLY
        # F-Right (types 0-1): still need phase, edge, transfer
        if self.phase_val is None:
            return SubActionPhase.CHOOSE_PHASE
        if self.new_edge is None:
            return SubActionPhase.CHOOSE_NEW_EDGE
        if self.transfer_edges is None:
            return SubActionPhase.CHOOSE_TRANSFER
        return SubActionPhase.APPLY

    @property
    def is_f_right(self) -> bool:
        return self.action_type is not None and self.action_type < 2

    def to_action_tuple(self) -> tuple:
        """Assemble the complete action tuple for env.step()."""
        graph_id = 0
        at = self.action_type
        ni = self.node_index
        ph = self.phase_val if self.phase_val is not None else 0
        ne = self.new_edge if self.new_edge is not None else 0
        te = self.transfer_edges if self.transfer_edges is not None else []
        return (graph_id, at, ni, ph, ne, *te)


# ---------------------------------------------------------------------------
# Annotated transition and trajectory (same interface as before)
# ---------------------------------------------------------------------------

@dataclass
class AnnotatedTransition:
    """A single sub-action transition with cached log-probabilities.

    In the decomposed design, each transition corresponds to ONE sub-decision
    (choose type, choose node, choose phase, etc.), not a full rewrite action.
    """
    sub_action_phase: SubActionPhase   # which sub-decision this was
    sub_action_value: object           # the chosen value (int, list, etc.)
    log_pf: torch.Tensor               # log P_F for this sub-decision
    log_pb: float                       # log P_B for this sub-decision


@dataclass
class AnnotatedTrajectory:
    """A complete trajectory with all data needed for TB loss computation.

    Transitions include both sub-action steps (within a single rewrite) and
    the implicit "apply rewrite" steps.  The TB loss sums over ALL transitions.
    """
    transitions: list[AnnotatedTransition] = field(default_factory=list)
    initial_t_gates: int = 0
    final_t_gates: int = 0
    terminal_reward: float = 0.0
    shaped_reward: float = 0.0  # terminal_reward × intermediate shaping bonus
    num_rewrites: int = 0   # number of actual ZX diagram rewrites applied
    pyzx_graph: object = None  # original PyZX graph for baseline comparison

    # Per-rewrite T-gate deltas for intermediate reward shaping.
    # delta > 0 means the rewrite reduced T-gates.
    per_rewrite_t_deltas: list[int] = field(default_factory=list)

    # Stored action tuples for replay (one per rewrite step).
    rewrite_actions: list[tuple] = field(default_factory=list)

    # Initial state for replay (only set when replay is enabled).
    initial_state: object = None  # GameState, typed as object to avoid circular import

    # Initial ZX diagram for cross-process replay.  GameState's internal
    # ZXMatchDiagram doesn't survive cross-process pickling, but the raw
    # ZXDiagram (a pyzx Graph) does.  When set, the main process can
    # reconstruct a fresh GameState via GameState.from_diagram().
    initial_diagram: object = None  # ZXDiagram, typed as object

    # Rewrite-boundary flow estimates from the model's value head.
    # rewrite_boundary_log_flows[k] = log F(s_k) for the ZX state BEFORE
    # rewrite k.  Used by SubTB(λ) loss for intermediate flow matching.
    # Length = num_rewrites (one per rewrite, at the pre-rewrite state).
    rewrite_boundary_log_flows: list = field(default_factory=list)

    # Index into self.transitions where each rewrite's sub-steps BEGIN.
    # rewrite_start_indices[k] is the transition index for the first
    # sub-step (CHOOSE_TYPE) of rewrite k.  Length = num_rewrites.
    rewrite_start_indices: list[int] = field(default_factory=list)

    # When this trajectory was produced by replaying a TrajectoryRecord
    # (parallel sampling), the original record is kept here so it can be
    # stored in the replay buffer and re-replayed via the tensor-based
    # path (which avoids hash-dependent match-diagram reconstruction).
    _source_record: object = None  # TrajectoryRecord

    @property
    def t_gate_reduction(self) -> int:
        return self.initial_t_gates - self.final_t_gates

    @property
    def sum_log_pf(self) -> torch.Tensor:
        """Sum of log P_F over all sub-action transitions (differentiable)."""
        if not self.transitions:
            return torch.tensor(0.0)
        # Squeeze to scalar — distribution methods return varying shapes
        return torch.stack([t.log_pf.squeeze() for t in self.transitions]).sum()

    @property
    def sum_log_pb(self) -> float:
        """Sum of log P_B over all sub-action transitions (scalar)."""
        return sum(t.log_pb for t in self.transitions)

    @property
    def action_type_fingerprint(self) -> tuple[int, ...]:
        """Tuple of action types used (for diversity scoring in replay)."""
        types = []
        for t in self.transitions:
            if t.sub_action_phase == SubActionPhase.CHOOSE_TYPE:
                types.append(t.sub_action_value)
        return tuple(types)

    def __len__(self) -> int:
        return len(self.transitions)


# ---------------------------------------------------------------------------
# Trajectory sampler — decomposed sub-actions
# ---------------------------------------------------------------------------

class TrajectorySampler:
    """Samples trajectories with decomposed sub-action transitions.

    Each rewrite action is broken into sequential sub-decisions:
      1. Choose action type  (log P_F = log P(type))
      2. Choose node         (log P_F = log P(node | type))
      3. [F-Right only] Choose phase        (log P_F = log P(phase | type, node))
      4. [F-Right only] Choose new_edge     (log P_F = log P(new_edge | type, node))
      5. [F-Right only] Choose transfer     (log P_F = log P(transfer | type, node))

    Each sub-decision becomes its own AnnotatedTransition in the trajectory.
    After all sub-decisions for one rewrite are made, the action is assembled
    and applied to produce the next ZX diagram.

    Backward policy (uniform):
      - Choose type:     log P_B = -log(num_available_types)
      - Choose node:     log P_B = -log(num_nodes_of_this_type)
      - Choose phase:    log P_B = -log(num_possible_phases)
      - Choose new_edge: log P_B = -log(num_possible_new_edges)
      - Choose transfer: log P_B = -log(2^num_transfer_dims)  [each bit independent]
    """

    def __init__(
        self,
        env: ZXGFlowNetEnv,
        policy: GFlowNetForwardPolicy,
        device: str = 'cpu',
        temperature: float = 1.0,
        epsilon_uniform: float = 0.0,
        max_trajectory_length: int = 20,
        reward_exponent: float = 4.0,
        min_reward: float = 0.01,
        reward_shaping_coeff: float = 0.0,
        retain_states_for_replay: bool = False,
    ):
        self.env = env
        self.policy = policy
        self.device = device
        self.temperature = temperature
        self.epsilon_uniform = epsilon_uniform
        self.max_length = max_trajectory_length
        self.reward_exponent = reward_exponent
        self.min_reward = min_reward
        self.reward_shaping_coeff = reward_shaping_coeff
        self.retain_states_for_replay = retain_states_for_replay

    @torch.no_grad()
    def sample(self) -> AnnotatedTrajectory:
        """Sample a single trajectory (no gradients — for data collection)."""
        return self._rollout(compute_grad=False)

    def sample_with_grad(self) -> AnnotatedTrajectory:
        """Sample a trajectory with gradients for on-policy TB training."""
        return self._rollout(compute_grad=True)

    def _rollout(self, compute_grad: bool = False,
                  retain_pyzx_graph: bool = False) -> AnnotatedTrajectory:
        """Run a full trajectory with decomposed sub-action transitions.

        Args:
            compute_grad: Whether to track gradients through forward policy.
            retain_pyzx_graph: If True, use ``generate_state_with_pyzx()``
                and store the original PyZX graph on the trajectory for
                baseline comparison.
        """
        if retain_pyzx_graph:
            state, pyzx_graph = self.env.generate_state_with_pyzx()
        else:
            state = self.env.generate_state()
            pyzx_graph = None
        initial_t_gates = state.num_non_clifford

        traj = AnnotatedTrajectory(
            initial_t_gates=initial_t_gates,
            pyzx_graph=pyzx_graph,
        )

        # Optionally store initial state/diagram for replay
        if self.retain_states_for_replay:
            traj.initial_state = state.clone()
            traj.initial_diagram = state.zx_diagram.copy()

        for _rewrite_step in range(self.max_length):
            if self.env.is_terminal(state):
                break

            # Enumerate available actions at this ZX diagram state
            flat_actions = self.env.enumerate_flat_actions(state)
            f_right_nodes = self.env.enumerate_f_right_nodes(state)

            if not flat_actions and not f_right_nodes:
                break

            # Run the model ONCE per rewrite to get the distribution
            if compute_grad:
                dist, state_flow = self.policy(state, self.device)
            else:
                with torch.no_grad():
                    dist, state_flow = self.policy(state, self.device)

            # Store rewrite boundary info for SubTB
            traj.rewrite_boundary_log_flows.append(state_flow.squeeze())
            traj.rewrite_start_indices.append(len(traj.transitions))

            # --- Sub-step 1: Choose action type ---
            available_types = set()
            for a in flat_actions:
                available_types.add(a[1])
            for at, _ni in f_right_nodes:
                available_types.add(at)
            available_types_list = sorted(available_types)

            action_type, log_pf_type = self._sample_type(
                dist, available_types_list, compute_grad,
            )
            log_pb_type = -math.log(len(available_types_list))

            traj.transitions.append(AnnotatedTransition(
                sub_action_phase=SubActionPhase.CHOOSE_TYPE,
                sub_action_value=action_type,
                log_pf=log_pf_type,
                log_pb=log_pb_type,
            ))

            # --- Sub-step 2: Choose node ---
            if action_type >= 2:
                # Non-F-Right: pick from flat actions of this type
                type_actions = [a for a in flat_actions if a[1] == action_type]
                node_indices = [a[2] for a in type_actions]
            else:
                # F-Right: pick from F-Right nodes of this type
                fr_of_type = [(at, ni) for at, ni in f_right_nodes if at == action_type]
                node_indices = [ni for _, ni in fr_of_type]

            node_index, log_pf_node = self._sample_node(
                dist, action_type, node_indices, compute_grad,
            )
            log_pb_node = -math.log(max(1, len(node_indices)))

            traj.transitions.append(AnnotatedTransition(
                sub_action_phase=SubActionPhase.CHOOSE_NODE,
                sub_action_value=node_index,
                log_pf=log_pf_node,
                log_pb=log_pb_node,
            ))

            # --- F-Right sub-steps 3-5: phase, new_edge, transfer ---
            partial = PartialAction(action_type=action_type, node_index=node_index)

            if action_type < 2:
                at_t = torch.tensor([[action_type]])
                n_t = torch.tensor([[node_index]])

                # Sub-step 3: phase
                phase_val, log_pf_phase = self._sample_phase(
                    dist, at_t, n_t, compute_grad,
                )
                # Uniform backward over all possible phases
                log_pb_phase = -math.log(_NUM_POSSIBLE_PHASES)

                traj.transitions.append(AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_PHASE,
                    sub_action_value=phase_val,
                    log_pf=log_pf_phase,
                    log_pb=log_pb_phase,
                ))
                partial.phase_val = phase_val

                # Sub-step 4: new_edge
                new_edge_val, log_pf_edge = self._sample_new_edge(
                    dist, at_t, n_t, compute_grad,
                )
                log_pb_edge = -math.log(_NUM_POSSIBLE_NEW_EDGES)

                traj.transitions.append(AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_NEW_EDGE,
                    sub_action_value=new_edge_val,
                    log_pf=log_pf_edge,
                    log_pb=log_pb_edge,
                ))
                partial.new_edge = new_edge_val

                # Sub-step 5: transfer_edges
                transfer_vals, log_pf_transfer = self._sample_transfer(
                    dist, at_t, n_t, compute_grad,
                )
                # Uniform backward: each transfer bit is binary, so 2^n possibilities
                n_transfer_dims = len(transfer_vals) if transfer_vals else 0
                log_pb_transfer = -n_transfer_dims * math.log(2) if n_transfer_dims > 0 else 0.0

                traj.transitions.append(AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_TRANSFER,
                    sub_action_value=transfer_vals,
                    log_pf=log_pf_transfer,
                    log_pb=log_pb_transfer,
                ))
                partial.transfer_edges = transfer_vals

            # --- Assemble and apply the rewrite ---
            action_tuple = partial.to_action_tuple()

            # Track T-gate count before rewrite
            t_before = state.num_non_clifford

            next_state, reward, done = self.env.step(state, action_tuple)
            traj.num_rewrites += 1

            # Track T-gate delta and store action for replay
            t_after = next_state.num_non_clifford
            traj.per_rewrite_t_deltas.append(t_before - t_after)
            traj.rewrite_actions.append(action_tuple)

            state = next_state
            if done:
                break

        # Record terminal state info
        traj.final_t_gates = state.num_non_clifford
        traj.terminal_reward = self.env.terminal_reward(
            initial_t_gates, traj.final_t_gates,
            self.reward_exponent, self.min_reward,
        )

        # Shaped reward (DEPRECATED — coeff should be 0.0; SubTB handles
        # intermediate credit via state flow estimates F(s_k)).
        # Shaped reward (DEPRECATED — coeff should be 0.0)
        if self.reward_shaping_coeff > 0 and initial_t_gates > 0:
            positive_reductions = sum(max(0, d) for d in traj.per_rewrite_t_deltas)
            shaping_bonus = self.reward_shaping_coeff * positive_reductions / initial_t_gates
            traj.shaped_reward = traj.terminal_reward * math.exp(shaping_bonus)
        else:
            traj.shaped_reward = traj.terminal_reward

        return traj

    # ------------------------------------------------------------------
    # Sub-action sampling helpers
    # ------------------------------------------------------------------

    def _sample_type(
        self, dist, available_types: list[int], compute_grad: bool,
    ) -> tuple[int, torch.Tensor]:
        """Sample an action type and return (type_idx, log_pf)."""
        mixture_probs = dist.mixture_dist_params
        if mixture_probs.dim() > 1:
            mixture_probs = mixture_probs.squeeze(0)

        num_types = mixture_probs.shape[0]
        mask = torch.zeros(num_types, dtype=torch.bool)
        for at in available_types:
            if at < num_types:
                mask[at] = True

        log_probs = torch.log(mixture_probs.clamp(min=1e-30))
        log_probs[~mask] = -1e9

        if self.temperature > 0 and self.temperature != 1.0:
            log_probs_scaled = log_probs / self.temperature
        else:
            log_probs_scaled = log_probs

        if self.epsilon_uniform > 0:
            probs = torch.softmax(log_probs_scaled, dim=-1)
            uniform = mask.float() / max(1, mask.sum().item())
            probs = (1 - self.epsilon_uniform) * probs + self.epsilon_uniform * uniform
            chosen = torch.multinomial(probs, 1).item()
            # Recompute log_pf from the original (un-mixed) distribution
            log_pf = dist.action_type_log_probs(torch.tensor([chosen]))
        else:
            probs = torch.softmax(log_probs_scaled, dim=-1)
            chosen = torch.multinomial(probs, 1).item()
            log_pf = dist.action_type_log_probs(torch.tensor([chosen]))

        if not compute_grad:
            log_pf = log_pf.detach()
        return chosen, log_pf

    def _sample_node(
        self, dist, action_type: int, node_indices: list[int], compute_grad: bool,
    ) -> tuple[int, torch.Tensor]:
        """Sample a node index and return (node_idx, log_pf)."""
        if len(node_indices) == 1:
            ni = node_indices[0]
            log_pf = dist.node_log_probs(
                torch.tensor([[action_type]]), torch.tensor([[ni]]),
            )
            if not compute_grad:
                log_pf = log_pf.detach()
            return ni, log_pf

        node_logits = []
        for ni in node_indices:
            lp = dist.node_log_probs(
                torch.tensor([[action_type]]), torch.tensor([[ni]]),
            )
            node_logits.append(lp.item())

        logits_t = torch.tensor(node_logits).clamp(min=-100)
        if self.temperature > 0 and self.temperature != 1.0:
            logits_t = logits_t / self.temperature
        probs = torch.softmax(logits_t, dim=-1)
        chosen_idx = torch.multinomial(probs, 1).item()
        ni = node_indices[chosen_idx]

        # log_pf from the model (not temperature-adjusted)
        log_pf = dist.node_log_probs(
            torch.tensor([[action_type]]), torch.tensor([[ni]]),
        )
        if not compute_grad:
            log_pf = log_pf.detach()
        return ni, log_pf

    def _sample_phase(
        self, dist, action_type_t, node_t, compute_grad: bool,
    ) -> tuple[int, torch.Tensor]:
        """Sample a phase and return (phase_val, log_pf)."""
        phases = dist.sample_phases(action_type_t, node_t)[0]
        phase_val = phases.squeeze().item()
        log_pf = dist.new_phase_log_probs(
            action_type_t, node_t, torch.tensor([[phase_val]]),
        )
        if not compute_grad:
            log_pf = log_pf.detach()
        return phase_val, log_pf

    def _sample_new_edge(
        self, dist, action_type_t, node_t, compute_grad: bool,
    ) -> tuple[int, torch.Tensor]:
        """Sample a new_edge and return (new_edge_val, log_pf)."""
        new_edges = dist.sample_new_edges(action_type_t, node_t)[0]
        edge_val = new_edges.squeeze().item()
        log_pf = dist.new_edge_log_probs(
            action_type_t, node_t, torch.tensor([[edge_val]]),
        )
        if not compute_grad:
            log_pf = log_pf.detach()
        return edge_val, log_pf

    def _sample_transfer(
        self, dist, action_type_t, node_t, compute_grad: bool,
    ) -> tuple[list[int], torch.Tensor]:
        """Sample transfer_edges and return (list_of_ints, log_pf)."""
        transfer = dist.sample_transfer_edges(action_type_t, node_t)[0]
        transfer_list = [int(x) for x in transfer.squeeze().tolist()]
        # Ensure it's a list even for scalar
        if isinstance(transfer_list, int):
            transfer_list = [transfer_list]
        transfer_t = torch.tensor([transfer_list], dtype=torch.float32)
        log_pf = dist.transfer_edge_log_probs(
            action_type_t, node_t, transfer_t,
        )
        if not compute_grad:
            log_pf = log_pf.detach()
        return transfer_list, log_pf

    # ------------------------------------------------------------------
    # Replay rollout — teacher-forced re-evaluation
    # ------------------------------------------------------------------

    def replay_rollout(
        self, initial_state: GameState, action_tuples: list[tuple],
    ) -> AnnotatedTrajectory:
        """Re-evaluate a stored trajectory under the current policy.

        Replays the exact sequence of actions, computing fresh log_pf
        values from the current model.  This provides proper gradients
        for replay-based training.

        Args:
            initial_state: The starting GameState (stored in replay buffer).
            action_tuples: The sequence of complete action tuples applied.

        Returns:
            AnnotatedTrajectory with current policy's log_pf values.
        """
        state = initial_state.clone()
        initial_t_gates = state.num_non_clifford

        traj = AnnotatedTrajectory(initial_t_gates=initial_t_gates)

        for action_tuple in action_tuples:
            if self.env.is_terminal(state):
                break

            # Parse the action tuple: (graph_id, action_type, node_index, phase, new_edge, *transfer)
            _graph_id = action_tuple[0]
            action_type = action_tuple[1]
            node_index = action_tuple[2]
            phase_val = action_tuple[3] if len(action_tuple) > 3 else 0
            new_edge_val = action_tuple[4] if len(action_tuple) > 4 else 0
            transfer_vals = list(action_tuple[5:]) if len(action_tuple) > 5 else []

            # Enumerate available actions to compute backward probabilities
            flat_actions = self.env.enumerate_flat_actions(state)
            f_right_nodes = self.env.enumerate_f_right_nodes(state)

            if not flat_actions and not f_right_nodes:
                break

            # Run the model to get current distribution
            with torch.no_grad() if not self.policy.training else torch.enable_grad():
                dist, state_flow = self.policy(state, self.device)

            # Store rewrite boundary info for SubTB
            traj.rewrite_boundary_log_flows.append(state_flow.squeeze())
            traj.rewrite_start_indices.append(len(traj.transitions))

            # --- Sub-step 1: Type (teacher-forced) ---
            available_types = set()
            for a in flat_actions:
                available_types.add(a[1])
            for at, _ni in f_right_nodes:
                available_types.add(at)

            log_pf_type = dist.action_type_log_probs(torch.tensor([action_type]))
            if not self.policy.training:
                log_pf_type = log_pf_type.detach()
            log_pb_type = -math.log(max(1, len(available_types)))

            traj.transitions.append(AnnotatedTransition(
                sub_action_phase=SubActionPhase.CHOOSE_TYPE,
                sub_action_value=action_type,
                log_pf=log_pf_type,
                log_pb=log_pb_type,
            ))

            # --- Sub-step 2: Node (teacher-forced) ---
            if action_type >= 2:
                type_actions = [a for a in flat_actions if a[1] == action_type]
                node_indices = [a[2] for a in type_actions]
            else:
                fr_of_type = [(at, ni) for at, ni in f_right_nodes if at == action_type]
                node_indices = [ni for _, ni in fr_of_type]

            log_pf_node = dist.node_log_probs(
                torch.tensor([[action_type]]), torch.tensor([[node_index]]),
            )
            if not self.policy.training:
                log_pf_node = log_pf_node.detach()
            log_pb_node = -math.log(max(1, len(node_indices)))

            traj.transitions.append(AnnotatedTransition(
                sub_action_phase=SubActionPhase.CHOOSE_NODE,
                sub_action_value=node_index,
                log_pf=log_pf_node,
                log_pb=log_pb_node,
            ))

            # --- F-Right sub-steps 3-5 (teacher-forced) ---
            if action_type < 2:
                at_t = torch.tensor([[action_type]])
                n_t = torch.tensor([[node_index]])

                # Phase
                log_pf_phase = dist.new_phase_log_probs(
                    at_t, n_t, torch.tensor([[phase_val]]),
                )
                if not self.policy.training:
                    log_pf_phase = log_pf_phase.detach()
                log_pb_phase = -math.log(_NUM_POSSIBLE_PHASES)
                traj.transitions.append(AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_PHASE,
                    sub_action_value=phase_val,
                    log_pf=log_pf_phase,
                    log_pb=log_pb_phase,
                ))

                # New edge
                log_pf_edge = dist.new_edge_log_probs(
                    at_t, n_t, torch.tensor([[new_edge_val]]),
                )
                if not self.policy.training:
                    log_pf_edge = log_pf_edge.detach()
                log_pb_edge = -math.log(_NUM_POSSIBLE_NEW_EDGES)
                traj.transitions.append(AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_NEW_EDGE,
                    sub_action_value=new_edge_val,
                    log_pf=log_pf_edge,
                    log_pb=log_pb_edge,
                ))

                # Transfer edges
                if transfer_vals:
                    transfer_t = torch.tensor([transfer_vals], dtype=torch.float32)
                    log_pf_transfer = dist.transfer_edge_log_probs(
                        at_t, n_t, transfer_t,
                    )
                    if not self.policy.training:
                        log_pf_transfer = log_pf_transfer.detach()
                    n_dims = len(transfer_vals)
                    log_pb_transfer = -n_dims * math.log(2) if n_dims > 0 else 0.0
                    traj.transitions.append(AnnotatedTransition(
                        sub_action_phase=SubActionPhase.CHOOSE_TRANSFER,
                        sub_action_value=transfer_vals,
                        log_pf=log_pf_transfer,
                        log_pb=log_pb_transfer,
                    ))

            # Apply the action
            t_before = state.num_non_clifford
            next_state, reward, done = self.env.step(state, action_tuple)
            traj.num_rewrites += 1

            t_after = next_state.num_non_clifford
            traj.per_rewrite_t_deltas.append(t_before - t_after)
            traj.rewrite_actions.append(action_tuple)

            state = next_state
            if done:
                break

        # Terminal info
        traj.final_t_gates = state.num_non_clifford
        traj.terminal_reward = self.env.terminal_reward(
            initial_t_gates, traj.final_t_gates,
            self.reward_exponent, self.min_reward,
        )

        # Shaped reward (DEPRECATED — coeff should be 0.0)
        if self.reward_shaping_coeff > 0 and initial_t_gates > 0:
            positive_reductions = sum(max(0, d) for d in traj.per_rewrite_t_deltas)
            shaping_bonus = self.reward_shaping_coeff * positive_reductions / initial_t_gates
            traj.shaped_reward = traj.terminal_reward * math.exp(shaping_bonus)
        else:
            traj.shaped_reward = traj.terminal_reward

        return traj

    # ------------------------------------------------------------------
    # Batch sampling
    # ------------------------------------------------------------------

    def sample_batch(self, n: int) -> list[AnnotatedTrajectory]:
        """Sample n trajectories (no gradients)."""
        results = []
        for i in range(n):
            results.append(self.sample())
            if n >= 4:
                logger.debug("  trajectory %d/%d sampled (%d sub-steps, %d rewrites)",
                             i + 1, n, len(results[-1]), results[-1].num_rewrites)
        return results

    def sample_batch_with_pyzx(self, n: int) -> list[AnnotatedTrajectory]:
        """Sample n trajectories (no grads) and retain PyZX graphs for baseline."""
        results = []
        for i in range(n):
            results.append(self._rollout(compute_grad=False, retain_pyzx_graph=True))
            if n >= 4:
                logger.debug("  trajectory %d/%d sampled (%d sub-steps, %d rewrites)",
                             i + 1, n, len(results[-1]), results[-1].num_rewrites)
        return results

    def rollout_from_state(
        self,
        state: 'GameState',
        pyzx_graph: object = None,
    ) -> AnnotatedTrajectory:
        """Run the agent on a *given* initial state (no gradients).

        This is used for benchmark evaluation where the circuit is fixed
        rather than randomly generated.  The rollout logic is identical to
        ``_rollout`` but skips state generation.

        Parameters
        ----------
        state : GameState
            The initial state to start from.  Will be cloned internally.
        pyzx_graph : pyzx.Graph, optional
            Original PyZX graph for baseline comparison.

        Returns
        -------
        AnnotatedTrajectory
        """
        state = state.clone()
        initial_t_gates = state.num_non_clifford

        traj = AnnotatedTrajectory(
            initial_t_gates=initial_t_gates,
            pyzx_graph=pyzx_graph,
        )

        for _rewrite_step in range(self.max_length):
            if self.env.is_terminal(state):
                break

            flat_actions = self.env.enumerate_flat_actions(state)
            f_right_nodes = self.env.enumerate_f_right_nodes(state)

            if not flat_actions and not f_right_nodes:
                break

            with torch.no_grad():
                dist, state_flow = self.policy(state, self.device)

            traj.rewrite_boundary_log_flows.append(state_flow.squeeze())
            traj.rewrite_start_indices.append(len(traj.transitions))

            # --- Sub-step 1: Choose action type ---
            available_types = set()
            for a in flat_actions:
                available_types.add(a[1])
            for at, _ni in f_right_nodes:
                available_types.add(at)
            available_types_list = sorted(available_types)

            action_type, log_pf_type = self._sample_type(
                dist, available_types_list, False,
            )
            log_pb_type = -math.log(len(available_types_list))

            traj.transitions.append(AnnotatedTransition(
                sub_action_phase=SubActionPhase.CHOOSE_TYPE,
                sub_action_value=action_type,
                log_pf=log_pf_type,
                log_pb=log_pb_type,
            ))

            # --- Sub-step 2: Choose node ---
            if action_type >= 2:
                type_actions = [a for a in flat_actions if a[1] == action_type]
                node_indices = [a[2] for a in type_actions]
            else:
                fr_of_type = [(at, ni) for at, ni in f_right_nodes if at == action_type]
                node_indices = [ni for _, ni in fr_of_type]

            node_index, log_pf_node = self._sample_node(
                dist, action_type, node_indices, False,
            )
            log_pb_node = -math.log(max(1, len(node_indices)))

            traj.transitions.append(AnnotatedTransition(
                sub_action_phase=SubActionPhase.CHOOSE_NODE,
                sub_action_value=node_index,
                log_pf=log_pf_node,
                log_pb=log_pb_node,
            ))

            # --- F-Right sub-steps 3-5 ---
            partial = PartialAction(action_type=action_type, node_index=node_index)

            if action_type < 2:
                at_t = torch.tensor([[action_type]])
                n_t = torch.tensor([[node_index]])

                phase_val, log_pf_phase = self._sample_phase(dist, at_t, n_t, False)
                log_pb_phase = -math.log(_NUM_POSSIBLE_PHASES)
                traj.transitions.append(AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_PHASE,
                    sub_action_value=phase_val,
                    log_pf=log_pf_phase,
                    log_pb=log_pb_phase,
                ))
                partial.phase_val = phase_val

                new_edge_val, log_pf_edge = self._sample_new_edge(dist, at_t, n_t, False)
                log_pb_edge = -math.log(_NUM_POSSIBLE_NEW_EDGES)
                traj.transitions.append(AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_NEW_EDGE,
                    sub_action_value=new_edge_val,
                    log_pf=log_pf_edge,
                    log_pb=log_pb_edge,
                ))
                partial.new_edge = new_edge_val

                transfer_vals, log_pf_transfer = self._sample_transfer(dist, at_t, n_t, False)
                n_transfer_dims = len(transfer_vals) if transfer_vals else 0
                log_pb_transfer = -n_transfer_dims * math.log(2) if n_transfer_dims > 0 else 0.0
                traj.transitions.append(AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_TRANSFER,
                    sub_action_value=transfer_vals,
                    log_pf=log_pf_transfer,
                    log_pb=log_pb_transfer,
                ))
                partial.transfer_edges = transfer_vals

            # --- Apply rewrite ---
            action_tuple = partial.to_action_tuple()
            t_before = state.num_non_clifford
            next_state, reward, done = self.env.step(state, action_tuple)
            traj.num_rewrites += 1
            t_after = next_state.num_non_clifford
            traj.per_rewrite_t_deltas.append(t_before - t_after)
            traj.rewrite_actions.append(action_tuple)

            state = next_state
            if done:
                break

        traj.final_t_gates = state.num_non_clifford
        traj.terminal_reward = self.env.terminal_reward(
            initial_t_gates, traj.final_t_gates,
            self.reward_exponent, self.min_reward,
        )
        traj.shaped_reward = traj.terminal_reward
        return traj

    def sample_batch_with_grad(self, n: int) -> list[AnnotatedTrajectory]:
        """Sample n trajectories with gradients."""
        results = []
        for i in range(n):
            results.append(self.sample_with_grad())
            if n >= 4:
                logger.debug("  trajectory %d/%d sampled (%d sub-steps, %d rewrites)",
                             i + 1, n, len(results[-1]), results[-1].num_rewrites)
        return results

    # ------------------------------------------------------------------
    # Replay from exported tensor records (parallel sampling)
    # ------------------------------------------------------------------

    def replay_from_records(
        self, record: TrajectoryRecord,
    ) -> AnnotatedTrajectory:
        """Replay a TrajectoryRecord through the current model with gradients.

        Unlike ``replay_rollout`` (which takes a GameState and re-runs the
        environment), this method takes pre-exported PyG Data tensors from
        a worker process and runs only the model forward pass.  No GameState,
        no environment interaction — just tensor → model → distribution → log_pf.

        This avoids the cross-process GameState pickling problem entirely:
        workers export preprocessed tensors, and the main process only needs
        to evaluate the model on them.

        Args:
            record: A TrajectoryRecord with RewriteStepRecords containing
                    preprocessed model-input tensors and action details.

        Returns:
            AnnotatedTrajectory with gradient-tracked log_pf values from
            the current model weights, ready for loss computation.
        """
        from alphazx.distributions.alpha_zx_dist import AlphaZXDistribution

        traj = AnnotatedTrajectory(
            initial_t_gates=record.initial_t_gates,
            final_t_gates=record.final_t_gates,
            terminal_reward=record.terminal_reward,
            shaped_reward=record.shaped_reward,
            num_rewrites=record.num_rewrites,
            per_rewrite_t_deltas=list(record.per_rewrite_t_deltas),
            rewrite_actions=list(record.rewrite_actions),
            initial_diagram=record.initial_diagram,
            _source_record=record,
        )

        for step in record.rewrite_steps:
            # --- Run model on exported tensors (WITH gradients) ---
            x = step.x.to(self.device)
            edge_index = step.edge_index.to(self.device)
            edge_attr = step.edge_attr.to(self.device)
            node_type = step.node_type.to(self.device)
            pe = step.pe.to(self.device)
            node_id = step.node_id.to(self.device)
            edge_type = (step.edge_type.to(self.device)
                         if step.edge_type is not None else None)

            batch_tensor = torch.zeros(
                x.shape[0], dtype=torch.long, device=self.device,
            )

            dist_params, state_flow = self.policy.model(
                x, edge_index, edge_attr, node_type,
                batch_tensor, pe, node_id,
                edge_type=edge_type,
            )
            distribution = AlphaZXDistribution(dist_params)

            # Store rewrite boundary info for SubTB
            traj.rewrite_boundary_log_flows.append(state_flow.squeeze())
            traj.rewrite_start_indices.append(len(traj.transitions))

            # --- Sub-step 1: Type (teacher-forced) ---
            log_pf_type = distribution.action_type_log_probs(
                torch.tensor([step.action_type]),
            )
            log_pb_type = -math.log(max(1, step.num_available_types))

            traj.transitions.append(AnnotatedTransition(
                sub_action_phase=SubActionPhase.CHOOSE_TYPE,
                sub_action_value=step.action_type,
                log_pf=log_pf_type,
                log_pb=log_pb_type,
            ))

            # --- Sub-step 2: Node (teacher-forced) ---
            log_pf_node = distribution.node_log_probs(
                torch.tensor([[step.action_type]]),
                torch.tensor([[step.node_index]]),
            )
            log_pb_node = -math.log(max(1, step.num_type_nodes))

            traj.transitions.append(AnnotatedTransition(
                sub_action_phase=SubActionPhase.CHOOSE_NODE,
                sub_action_value=step.node_index,
                log_pf=log_pf_node,
                log_pb=log_pb_node,
            ))

            # --- F-Right sub-steps 3-5 (teacher-forced) ---
            if step.action_type < 2:
                at_t = torch.tensor([[step.action_type]])
                n_t = torch.tensor([[step.node_index]])

                # Phase
                log_pf_phase = distribution.new_phase_log_probs(
                    at_t, n_t, torch.tensor([[step.phase_val]]),
                )
                log_pb_phase = -math.log(_NUM_POSSIBLE_PHASES)
                traj.transitions.append(AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_PHASE,
                    sub_action_value=step.phase_val,
                    log_pf=log_pf_phase,
                    log_pb=log_pb_phase,
                ))

                # New edge
                log_pf_edge = distribution.new_edge_log_probs(
                    at_t, n_t, torch.tensor([[step.new_edge_val]]),
                )
                log_pb_edge = -math.log(_NUM_POSSIBLE_NEW_EDGES)
                traj.transitions.append(AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_NEW_EDGE,
                    sub_action_value=step.new_edge_val,
                    log_pf=log_pf_edge,
                    log_pb=log_pb_edge,
                ))

                # Transfer edges
                if step.transfer_vals:
                    transfer_t = torch.tensor(
                        [step.transfer_vals], dtype=torch.float32,
                    )
                    log_pf_transfer = distribution.transfer_edge_log_probs(
                        at_t, n_t, transfer_t,
                    )
                    n_dims = step.num_transfer_dims
                    log_pb_transfer = (
                        -n_dims * math.log(2) if n_dims > 0 else 0.0
                    )
                    traj.transitions.append(AnnotatedTransition(
                        sub_action_phase=SubActionPhase.CHOOSE_TRANSFER,
                        sub_action_value=step.transfer_vals,
                        log_pf=log_pf_transfer,
                        log_pb=log_pb_transfer,
                    ))

        return traj

    def replay_records_batched(
        self, records: list[TrajectoryRecord],
    ) -> list[AnnotatedTrajectory]:
        """Replay multiple TrajectoryRecords with a single batched forward pass.

        This is a drop-in replacement for calling ``replay_from_records`` in
        a loop.  Instead of running the model N times (once per rewrite step
        across all trajectories), it:

        1. Collects all rewrite-step tensors into a flat list of PyG Data.
        2. Packs them into a single ``Batch`` with ``Batch.from_data_list``.
        3. Runs **one** batched model forward pass.
        4. Slices the distribution params back per-step and teacher-forces
           each sub-action to compute gradient-tracked ``log_pf`` values.

        The batched forward pass amortises GNN message-passing overhead across
        all steps, giving a substantial speedup when there are many rewrite
        steps (typical: 5-15 rewrites × 8-16 trajectories = 40-240 graphs).

        Gradient flow is preserved: the per-step ``log_pf`` tensors are
        autograd-connected to the batched model output via tensor slicing.

        Args:
            records: List of TrajectoryRecords from parallel workers.

        Returns:
            List of AnnotatedTrajectory (same order as *records*), each with
            gradient-tracked log_pf values from the current model weights.
        """
        from alphazx.distributions.alpha_zx_dist import (
            AlphaZXDistribution, AlphaZXDistributionParams,
        )
        from torch_geometric.data import Data, Batch

        # --- 1. Build flat list of (record_idx, step) pairs and Data objects ---
        step_index: list[tuple[int, RewriteStepRecord]] = []
        data_list: list[Data] = []

        for rec_idx, record in enumerate(records):
            for step in record.rewrite_steps:
                step_index.append((rec_idx, step))
                d = Data(
                    x=step.x,
                    edge_index=step.edge_index,
                    edge_attr=step.edge_attr,
                    node_type=step.node_type,
                    pe=step.pe,
                    id=step.node_id,
                )
                if step.edge_type is not None:
                    d.edge_type = step.edge_type
                data_list.append(d)

        # --- Initialise AnnotatedTrajectory shells for each record ---
        trajectories: list[AnnotatedTrajectory] = []
        for record in records:
            traj = AnnotatedTrajectory(
                initial_t_gates=record.initial_t_gates,
                final_t_gates=record.final_t_gates,
                terminal_reward=record.terminal_reward,
                shaped_reward=record.shaped_reward,
                num_rewrites=record.num_rewrites,
                per_rewrite_t_deltas=list(record.per_rewrite_t_deltas),
                rewrite_actions=list(record.rewrite_actions),
                initial_diagram=record.initial_diagram,
                _source_record=record,
            )
            trajectories.append(traj)

        if not data_list:
            return trajectories

        # --- 2. Batch all graphs and run a single model forward pass ---
        batch = Batch.from_data_list(data_list).to(self.device)
        graph_ids = torch.stack([d.id for d in data_list]).to(self.device)

        dist_params_all, state_flows_all = self.policy.model(
            batch.x, batch.edge_index, batch.edge_attr,
            batch.node_type,
            batch.batch,
            batch.pe,
            graph_ids,
            edge_type=getattr(batch, 'edge_type', None),
        )

        # --- 3. Slice per-step and teacher-force sub-actions ---
        for batch_idx, (rec_idx, step) in enumerate(step_index):
            traj = trajectories[rec_idx]

            single_params = AlphaZXDistributionParams(
                graph_ids=dist_params_all.graph_ids[batch_idx:batch_idx + 1],
                mixture_dist_probs=dist_params_all.mixture_dist_probs[batch_idx:batch_idx + 1],
                node_dist_probs=dist_params_all.node_dist_probs[batch_idx:batch_idx + 1],
                phase_dist_probs=dist_params_all.phase_dist_probs[batch_idx:batch_idx + 1],
                new_edge_dist_probs=dist_params_all.new_edge_dist_probs[batch_idx:batch_idx + 1],
                transfer_edge_dist_probs=dist_params_all.transfer_edge_dist_probs[batch_idx:batch_idx + 1],
            )
            distribution = AlphaZXDistribution(single_params)

            # Rewrite boundary info for SubTB
            traj.rewrite_boundary_log_flows.append(
                state_flows_all[batch_idx].squeeze(),
            )
            traj.rewrite_start_indices.append(len(traj.transitions))

            # --- Sub-step 1: Type (teacher-forced) ---
            log_pf_type = distribution.action_type_log_probs(
                torch.tensor([step.action_type]),
            )
            log_pb_type = -math.log(max(1, step.num_available_types))

            traj.transitions.append(AnnotatedTransition(
                sub_action_phase=SubActionPhase.CHOOSE_TYPE,
                sub_action_value=step.action_type,
                log_pf=log_pf_type,
                log_pb=log_pb_type,
            ))

            # --- Sub-step 2: Node (teacher-forced) ---
            log_pf_node = distribution.node_log_probs(
                torch.tensor([[step.action_type]]),
                torch.tensor([[step.node_index]]),
            )
            log_pb_node = -math.log(max(1, step.num_type_nodes))

            traj.transitions.append(AnnotatedTransition(
                sub_action_phase=SubActionPhase.CHOOSE_NODE,
                sub_action_value=step.node_index,
                log_pf=log_pf_node,
                log_pb=log_pb_node,
            ))

            # --- F-Right sub-steps 3-5 (teacher-forced) ---
            if step.action_type < 2:
                at_t = torch.tensor([[step.action_type]])
                n_t = torch.tensor([[step.node_index]])

                # Phase
                log_pf_phase = distribution.new_phase_log_probs(
                    at_t, n_t, torch.tensor([[step.phase_val]]),
                )
                log_pb_phase = -math.log(_NUM_POSSIBLE_PHASES)
                traj.transitions.append(AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_PHASE,
                    sub_action_value=step.phase_val,
                    log_pf=log_pf_phase,
                    log_pb=log_pb_phase,
                ))

                # New edge
                log_pf_edge = distribution.new_edge_log_probs(
                    at_t, n_t, torch.tensor([[step.new_edge_val]]),
                )
                log_pb_edge = -math.log(_NUM_POSSIBLE_NEW_EDGES)
                traj.transitions.append(AnnotatedTransition(
                    sub_action_phase=SubActionPhase.CHOOSE_NEW_EDGE,
                    sub_action_value=step.new_edge_val,
                    log_pf=log_pf_edge,
                    log_pb=log_pb_edge,
                ))

                # Transfer edges
                if step.transfer_vals:
                    transfer_t = torch.tensor(
                        [step.transfer_vals], dtype=torch.float32,
                    )
                    log_pf_transfer = distribution.transfer_edge_log_probs(
                        at_t, n_t, transfer_t,
                    )
                    n_dims = step.num_transfer_dims
                    log_pb_transfer = (
                        -n_dims * math.log(2) if n_dims > 0 else 0.0
                    )
                    traj.transitions.append(AnnotatedTransition(
                        sub_action_phase=SubActionPhase.CHOOSE_TRANSFER,
                        sub_action_value=step.transfer_vals,
                        log_pf=log_pf_transfer,
                        log_pb=log_pb_transfer,
                    ))

        return trajectories


# ---------------------------------------------------------------------------
# Serializable trajectory data for parallel sampling
# ---------------------------------------------------------------------------

@dataclass
class RewriteStepRecord:
    """Pickle-safe record of one rewrite step for cross-process replay.

    Contains the preprocessed PyG Data (just tensors, no graph objects)
    and all action/backward-probability info needed to compute log_pf
    and log_pb in the main process without touching the environment.
    """
    # Preprocessed model inputs (detached tensors, pickle-safe)
    x: torch.Tensor                # node features [N, D]
    edge_index: torch.Tensor       # [2, E]
    edge_attr: torch.Tensor        # [E, ...]
    node_type: torch.Tensor        # [N]
    pe: torch.Tensor               # [N, pe_dim]
    node_id: torch.Tensor          # [N]
    edge_type: object              # [E] or None (hetero model)

    # Action details
    action_type: int
    node_index: int
    phase_val: int
    new_edge_val: int
    transfer_vals: list[int]

    # Backward probability counts
    num_available_types: int
    num_type_nodes: int
    num_transfer_dims: int

    # T-gate tracking
    t_gates_before: int


@dataclass
class TrajectoryRecord:
    """Pickle-safe record of a complete sampled trajectory.

    Contains all information needed by the main process to compute
    log_pf values under the current model via ``replay_from_records()``.
    No GameState, no ZXMatchDiagram — just tensors and plain Python data.

    Architecture note: GameState's internal ZXMatchDiagram uses Python
    sets whose iteration order depends on PYTHONHASHSEED. Since worker
    processes get different hash seeds, GameState can't be reliably
    reconstructed or pickled across processes.  We sidestep this by
    exporting preprocessed PyG Data tensors at each rewrite boundary.

    The ``initial_diagram`` field stores the raw ZXDiagram (an
    ``nx.MultiGraph`` subclass) which *does* survive pickling.  The main
    process can reconstruct a GameState via ``GameState.from_diagram()``
    for replay buffer admission.
    """
    rewrite_steps: list[RewriteStepRecord]   # one per rewrite
    rewrite_actions: list[tuple]             # complete action tuples
    initial_t_gates: int
    final_t_gates: int
    terminal_reward: float
    shaped_reward: float
    num_rewrites: int
    per_rewrite_t_deltas: list[int]
    action_type_fingerprint: tuple[int, ...]
    initial_diagram: object = None  # ZXDiagram for replay buffer admission


# ---------------------------------------------------------------------------
# Module-level worker function (required for ProcessPoolExecutor pickling)
# ---------------------------------------------------------------------------

def _worker_sample_trajectories(
    model_state_dict: dict,
    model_hparams: dict,
    sampler_kwargs: dict,
    num_trajectories: int,
    worker_seed: int,
) -> list[TrajectoryRecord]:
    """Sample GFlowNet trajectories in a worker process (no gradients).

    This is a module-level function (required for pickling by
    ProcessPoolExecutor).  Each invocation:
    1. Seeds RNGs for reproducibility/diversity.
    2. Reconstructs the model + policy, loads state_dict.
    3. Samples trajectories, exporting preprocessed PyG Data tensors
       at each rewrite boundary into RewriteStepRecord objects.
    4. Returns TrajectoryRecord objects (pickle-safe, no GameState).

    Architecture note:
    Workers export preprocessed model-input tensors rather than GameState
    objects because GameState's ZXMatchDiagram uses Python sets whose
    iteration order varies across processes (PYTHONHASHSEED).  The main
    process only needs the exported tensors to run the model forward pass
    and compute gradient-tracked log_pf values.

    :param model_state_dict: Serialized model weights (dict of CPU tensors).
    :param model_hparams: Dict of constructor args for the model.
    :param sampler_kwargs: Dict of TrajectorySampler constructor kwargs
                           (minus env/policy/device which are built locally).
    :param num_trajectories: Number of trajectories to sample.
    :param worker_seed: Seed for this worker's RNGs.
    :return: List of TrajectoryRecord objects.
    """
    # --- Prevent thread oversubscription ---
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
    torch.set_num_threads(1)

    # Seed RNGs
    torch.manual_seed(worker_seed)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2**32))

    # Suppress noisy logging in workers
    logging.getLogger('alphazx').setLevel(logging.WARNING)
    _wlog = logging.getLogger(f'{__name__}.worker')
    _wlog.setLevel(logging.INFO)
    if not _wlog.handlers:
        _h = logging.StreamHandler(sys.stderr)
        _h.setFormatter(logging.Formatter(
            '%(asctime)s [gfn-worker-%(process)d] %(message)s', datefmt='%H:%M:%S',
        ))
        _wlog.addHandler(_h)

    t0 = time.time()

    # Reconstruct model (deferred import to avoid circular deps)
    from alphazx.mcts.parallel_self_play import _build_model_from_hparams
    from alphazx.shared.evaluate import _preprocess_data_for_model

    model = _build_model_from_hparams(model_hparams)
    model.load_state_dict(model_state_dict)
    model.eval()

    # Build environment and policy
    from alphazx.shared.config import CircuitConfig
    env_config = CircuitConfig(**sampler_kwargs.pop('env_config'))
    env = ZXGFlowNetEnv(env_config)
    pe_dim = sampler_kwargs.pop('pe_dim', 20)
    policy = GFlowNetForwardPolicy(model, pe_dim=pe_dim)

    # Build sampler
    sampler = TrajectorySampler(
        env=env,
        policy=policy,
        device='cpu',
        retain_states_for_replay=False,  # we export tensors instead
        **sampler_kwargs,
    )

    # Sample trajectories and build RewriteStepRecords with exported tensors
    records: list[TrajectoryRecord] = []
    for _i in range(num_trajectories):
        state = env.generate_state()
        initial_t_gates = state.num_non_clifford
        # Store a copy of the initial diagram for replay buffer admission.
        # ZXDiagram (nx.MultiGraph subclass) is pickle-safe, unlike GameState.
        initial_diagram = state.zx_diagram.copy()

        rewrite_steps: list[RewriteStepRecord] = []
        rewrite_actions: list[tuple] = []
        per_rewrite_t_deltas: list[int] = []
        action_types_seen: list[int] = []

        for _step in range(sampler.max_length):
            if env.is_terminal(state):
                break

            flat_actions = env.enumerate_flat_actions(state)
            f_right_nodes = env.enumerate_f_right_nodes(state)
            if not flat_actions and not f_right_nodes:
                break

            # --- Export preprocessed PyG Data tensors ---
            data = state.data.clone()
            data = _preprocess_data_for_model(data, pe_dim)

            # --- Run model (no grad) to sample action ---
            with torch.no_grad():
                dist, _state_flow = policy(state, 'cpu')

            # --- Sub-step 1: Choose type ---
            available_types = set()
            for a in flat_actions:
                available_types.add(a[1])
            for at, _ni in f_right_nodes:
                available_types.add(at)
            available_types_list = sorted(available_types)

            action_type, _log_pf = sampler._sample_type(
                dist, available_types_list, False,
            )

            # --- Sub-step 2: Choose node ---
            if action_type >= 2:
                type_actions = [a for a in flat_actions if a[1] == action_type]
                node_indices = [a[2] for a in type_actions]
            else:
                fr_of_type = [(at, ni) for at, ni in f_right_nodes
                              if at == action_type]
                node_indices = [ni for _, ni in fr_of_type]

            node_index, _log_pf = sampler._sample_node(
                dist, action_type, node_indices, False,
            )

            # --- F-Right sub-steps ---
            partial = PartialAction(action_type=action_type,
                                    node_index=node_index)
            phase_val = 0
            new_edge_val = 0
            transfer_vals: list[int] = []
            n_transfer_dims = 0

            if action_type < 2:
                at_t = torch.tensor([[action_type]])
                n_t = torch.tensor([[node_index]])

                phase_val, _ = sampler._sample_phase(dist, at_t, n_t, False)
                partial.phase_val = phase_val

                new_edge_val, _ = sampler._sample_new_edge(dist, at_t, n_t, False)
                partial.new_edge = new_edge_val

                transfer_vals, _ = sampler._sample_transfer(dist, at_t, n_t, False)
                partial.transfer_edges = transfer_vals
                n_transfer_dims = len(transfer_vals) if transfer_vals else 0

            # --- Build RewriteStepRecord with exported tensors ---
            step_record = RewriteStepRecord(
                x=data.x.detach().cpu(),
                edge_index=data.edge_index.detach().cpu(),
                edge_attr=data.edge_attr.detach().cpu(),
                node_type=data.node_type.detach().cpu(),
                pe=data.pe.detach().cpu(),
                node_id=data.id.detach().cpu(),
                edge_type=(data.edge_type.detach().cpu()
                           if hasattr(data, 'edge_type')
                              and data.edge_type is not None
                           else None),
                action_type=action_type,
                node_index=node_index,
                phase_val=phase_val,
                new_edge_val=new_edge_val,
                transfer_vals=transfer_vals,
                num_available_types=len(available_types_list),
                num_type_nodes=max(1, len(node_indices)),
                num_transfer_dims=n_transfer_dims,
                t_gates_before=state.num_non_clifford,
            )
            rewrite_steps.append(step_record)
            action_types_seen.append(action_type)

            # --- Apply the rewrite ---
            action_tuple = partial.to_action_tuple()
            rewrite_actions.append(action_tuple)

            t_before = state.num_non_clifford
            next_state, reward, done = env.step(state, action_tuple)
            t_after = next_state.num_non_clifford
            per_rewrite_t_deltas.append(t_before - t_after)

            state = next_state
            if done:
                break

        # Terminal reward
        final_t_gates = state.num_non_clifford
        terminal_reward = env.terminal_reward(
            initial_t_gates, final_t_gates,
            sampler.reward_exponent, sampler.min_reward,
        )

        # Shaped reward (DEPRECATED — coeff should be 0.0)
        if sampler.reward_shaping_coeff > 0 and initial_t_gates > 0:
            positive_reductions = sum(max(0, d) for d in per_rewrite_t_deltas)
            shaping_bonus = (sampler.reward_shaping_coeff
                             * positive_reductions / initial_t_gates)
            shaped_reward = terminal_reward * math.exp(shaping_bonus)
        else:
            shaped_reward = terminal_reward

        records.append(TrajectoryRecord(
            rewrite_steps=rewrite_steps,
            rewrite_actions=rewrite_actions,
            initial_t_gates=initial_t_gates,
            final_t_gates=final_t_gates,
            terminal_reward=terminal_reward,
            shaped_reward=shaped_reward,
            num_rewrites=len(rewrite_steps),
            per_rewrite_t_deltas=per_rewrite_t_deltas,
            action_type_fingerprint=tuple(action_types_seen),
            initial_diagram=initial_diagram,
        ))

    return records


# ---------------------------------------------------------------------------
# Parallel trajectory sampler manager
# ---------------------------------------------------------------------------

class ParallelTrajectorySampler:
    """Orchestrates multi-process trajectory sampling for GFlowNet training.

    Architecture (sample-then-replay):
    1. Workers sample trajectories without gradients using frozen model
       weights.  Each worker builds its own model/env/policy in-process.
    2. Workers return TrajectoryRecord objects (pickle-safe).
    3. Main process replays recorded actions through the current model
       with gradients, producing AnnotatedTrajectory objects ready for
       loss computation.

    The ProcessPoolExecutor is created once and reused across iterations.

    Usage::

        parallel_sampler = ParallelTrajectorySampler(
            model=model,
            config=config,
            sampler=main_process_sampler,
            num_workers=4,
        )
        trajectories = parallel_sampler.sample_and_replay(n=64)
        # trajectories are AnnotatedTrajectory with gradients
    """

    def __init__(
        self,
        model: nn.Module,
        config,  # GFlowNetConfig
        sampler: TrajectorySampler,
        num_workers: int = 4,
    ):
        self.model = model
        self.config = config
        self.sampler = sampler  # used for replay in main process
        self.num_workers = num_workers

        # Extract model hparams once (reuses MCTS utility)
        from alphazx.mcts.parallel_self_play import _extract_model_hparams
        self._model_hparams = _extract_model_hparams(model)

        # Build sampler kwargs that workers need (everything except
        # env/policy/device which they build locally)
        self._sampler_kwargs = {
            'temperature': config.sampling_temperature,
            'epsilon_uniform': config.epsilon_uniform,
            'max_trajectory_length': config.max_trajectory_length,
            'reward_exponent': config.reward_exponent,
            'min_reward': config.min_reward,
            'reward_shaping_coeff': config.reward_shaping_coeff,
            'pe_dim': config.pe_dim,
            'env_config': {
                'num_qubits': config.num_qubits,
                'depth': config.depth,
                'circuit_type': config.circuit_type,
                'p_had': config.p_had,
                'p_t': config.p_t,
                'min_initial_t_gates': config.min_initial_t_gates,
                'max_t_gate_increase': config.max_t_gate_increase,
                'max_episode_length': config.max_episode_length,
                'pe_dim': config.pe_dim,
            },
        }

        # Thread-limiting env vars before spawning workers
        if num_workers > 1:
            os.environ['OMP_NUM_THREADS'] = '1'
            os.environ['MKL_NUM_THREADS'] = '1'
            os.environ['OPENBLAS_NUM_THREADS'] = '1'
            os.environ['VECLIB_MAXIMUM_THREADS'] = '1'

        self._executor = ProcessPoolExecutor(
            max_workers=num_workers,
            mp_context=mp.get_context('spawn'),
        )

        # Lifetime stats
        self.total_trajectories_sampled: int = 0
        self.total_sample_time: float = 0.0
        self.total_replay_time: float = 0.0

    def sample_and_replay(
        self, n: int, reward_exponent: float | None = None,
    ) -> list[AnnotatedTrajectory]:
        """Sample n trajectories in parallel, then replay with gradients.

        Args:
            n: Number of trajectories to sample.
            reward_exponent: Override reward exponent for this batch
                (used for annealing).  If None, uses config default.

        Returns:
            List of AnnotatedTrajectory with gradient-tracked log_pf
            values from the current model weights.
        """
        t_sample_start = time.time()

        # Serialize current model weights
        state_dict = {k: v.cpu() for k, v in self.model.state_dict().items()}

        # Build sampler kwargs with current reward exponent
        kwargs = dict(self._sampler_kwargs)
        if reward_exponent is not None:
            kwargs['reward_exponent'] = reward_exponent

        # Partition trajectories across workers
        per_worker = self._partition(n)
        base_seed = int(time.time() * 1000) % (2**31)

        # Dispatch to workers
        futures = []
        for i, n_traj in enumerate(per_worker):
            if n_traj == 0:
                continue
            # Deep-copy kwargs so workers don't share mutable dicts
            worker_kwargs = {k: (dict(v) if isinstance(v, dict) else v)
                            for k, v in kwargs.items()}
            future = self._executor.submit(
                _worker_sample_trajectories,
                model_state_dict=state_dict,
                model_hparams=self._model_hparams,
                sampler_kwargs=worker_kwargs,
                num_trajectories=n_traj,
                worker_seed=base_seed + i * 10_000,
            )
            futures.append(future)

        # Collect results
        all_records: list[TrajectoryRecord] = []
        for idx, future in enumerate(futures):
            try:
                records = future.result()
                all_records.extend(records)
            except Exception as e:
                logger.error(f"GFlowNet worker {idx+1} failed: {e}")

        t_sample_elapsed = time.time() - t_sample_start
        self.total_sample_time += t_sample_elapsed
        self.total_trajectories_sampled += len(all_records)

        logger.info(
            f"Parallel sampling: {len(all_records)}/{n} trajectories in "
            f"{t_sample_elapsed:.1f}s across {self.num_workers} workers"
        )

        # Replay through current model with gradients
        t_replay_start = time.time()
        trajectories = self._replay_records(all_records)
        t_replay_elapsed = time.time() - t_replay_start
        self.total_replay_time += t_replay_elapsed

        logger.info(
            f"Replay: {len(trajectories)} trajectories in {t_replay_elapsed:.1f}s"
        )

        return trajectories

    def _replay_records(
        self, records: list[TrajectoryRecord],
    ) -> list[AnnotatedTrajectory]:
        """Replay trajectory records through the current model with gradients.

        Uses a single batched forward pass across ALL rewrite steps from
        ALL trajectories, then slices the results to teacher-force each
        sub-action.  This replaces the sequential per-record approach and
        gives a large speedup by amortising GNN message-passing overhead.

        Records with no rewrite steps are still included in the output
        (as empty trajectories) to preserve alignment with the input list.
        """
        # Filter out empty records but track indices for alignment
        non_empty = [r for r in records if r.rewrite_steps]
        if not non_empty:
            return [
                AnnotatedTrajectory(
                    initial_t_gates=r.initial_t_gates,
                    final_t_gates=r.final_t_gates,
                    terminal_reward=r.terminal_reward,
                    shaped_reward=r.shaped_reward,
                    num_rewrites=r.num_rewrites,
                    per_rewrite_t_deltas=list(r.per_rewrite_t_deltas),
                    rewrite_actions=list(r.rewrite_actions),
                    initial_diagram=r.initial_diagram,
                    _source_record=r,
                )
                for r in records
            ]
        return self.sampler.replay_records_batched(non_empty)

    def _partition(self, n: int) -> list[int]:
        """Partition n trajectories across workers."""
        base = n // self.num_workers
        remainder = n % self.num_workers
        return [
            base + (1 if i < remainder else 0)
            for i in range(self.num_workers)
        ]

    def shutdown(self) -> None:
        """Shutdown the process pool."""
        self._executor.shutdown(wait=True)

    def __del__(self):
        try:
            self._executor.shutdown(wait=False)
        except Exception:
            pass
