"""
GameState: a lightweight, copyable snapshot of a ZX diagram for use in MCTS tree search.

Key design decisions:
- GameState wraps a ZXDiagram + ZXMatchDiagram + PyG Data together.
- clone() creates an independent copy that can be mutated without affecting the original.
- apply_action() mutates this state in place (caller is responsible for cloning first).
- The PyG Data and DataIndexToMatch are lazily recomputed after each action.

This class does NOT own episode-level bookkeeping (cumulative reward, episode length, etc.).
That belongs in the self-play loop, not in the search tree.
"""

from __future__ import annotations

import networkx as nx
from torch_geometric.data import Data

from alphazx.diagram.match import MatchNode
from alphazx.diagram.zx_diagram import ZXDiagram
from alphazx.diagram.zx_match_diagram import ZXMatchDiagram, to_zx_match_diagram, DataIndexToMatch
from alphazx.game.zx_game import (
    num_non_clifford_gates,
    is_simplified,
    remove_isolated_nodes,
    remove_self_loop_edges,
    remove_isolated_components,
    update_match_diagram_for_removed_nodes,
    tuple_to_match,
    DiagramStats,
    calculate_reward,
)
from alphazx.rewriting.efficient_rewrite import efficient_rewrite

from alphazx.diagram.match import METADATA


def _clone_match_diagram(src: ZXMatchDiagram, new_zx_diagram: ZXDiagram) -> ZXMatchDiagram:
    """
    Create an independent shallow copy of a ZXMatchDiagram, bypassing __init__.

    This copies the underlying nx.DiGraph data (nodes, edges, attribute dicts)
    directly, then copies the custom instance attributes (type-specific node sets,
    super_nodes, etc.) and re-points the zx_diagram reference to `new_zx_diagram`.

    This is O(V + E) in the match diagram size — far cheaper than recomputing
    all matches from scratch via to_zx_match_diagram(), which involves pattern
    detection across the entire ZX diagram.
    """
    # Create an empty ZXMatchDiagram-shaped object without calling __init__.
    # object.__new__ skips __init__ entirely.
    clone = object.__new__(ZXMatchDiagram)

    # Initialize the nx.DiGraph base class directly
    nx.DiGraph.__init__(clone)

    # Copy graph-level attributes
    clone.graph.update(src.graph)

    # Copy nodes with shallow-copied attribute dicts.
    # Use nx.DiGraph methods directly to bypass ZXMatchDiagram.add_node, which
    # would recompute attributes and double-add to type-specific sets.
    nx.DiGraph.add_nodes_from(clone, ((n, d.copy()) for n, d in src._node.items()))

    # Copy edges with shallow-copied attribute dicts.
    # Use nx.DiGraph.add_edges_from to bypass ZXMatchDiagram.add_edge, which
    # would recompute edge attributes and add reverse edges (already present).
    nx.DiGraph.add_edges_from(
        clone,
        ((u, v, datadict.copy())
         for u, nbrs in src._adj.items()
         for v, datadict in nbrs.items())
    )

    # Copy ZXMatchDiagram-specific instance attributes
    clone.zx_diagram = new_zx_diagram
    clone.phase_denominator = new_zx_diagram.phase_denominator
    clone.node_attrs = new_zx_diagram.node_attrs
    clone.edge_attrs = new_zx_diagram.edge_attrs
    clone.super_nodes = set(src.super_nodes)

    # Copy all type-specific node sets (frz_nodes, frx_nodes, flz_nodes, etc.)
    for abbrev in METADATA.match_node_type_abbrevs:
        attr_name = f'{abbrev}_nodes'
        setattr(clone, attr_name, set(getattr(src, attr_name)))

    return clone


class GameState:
    """
    An immutable-by-convention snapshot of a ZX game state for MCTS.

    Usage in MCTS:
        child_state = parent_state.clone()
        reward, done = child_state.apply_action(action)
    """

    def __init__(self, zx_diagram: ZXDiagram, zx_match_diagram: ZXMatchDiagram):
        self.zx_diagram = zx_diagram
        self.zx_match_diagram = zx_match_diagram
        # Lazily computed — call ensure_data() before accessing
        self._data: Data | None = None
        self._data_index: DataIndexToMatch | None = None
        self._diagram_stats: DiagramStats | None = None
        # Cache for preprocessed data from evaluate_state, reused by _preprocess_state
        self._cached_preprocessed_data: Data | None = None
        # Cached non-Clifford gate count (invalidated by apply_action)
        self._num_non_clifford: int | None = None

    @classmethod
    def from_diagram(cls, zx_diagram: ZXDiagram) -> GameState:
        """Create a GameState from a ZXDiagram, computing the match diagram from scratch."""
        remove_isolated_nodes(zx_diagram)
        remove_self_loop_edges(zx_diagram)
        remove_isolated_components(zx_diagram)
        zx_match_diagram = to_zx_match_diagram(zx_diagram)
        return cls(zx_diagram, zx_match_diagram)

    @classmethod
    def from_game(cls, game) -> GameState:
        """Create a GameState from an existing ZXGame, sharing (not copying) the diagram."""
        state = cls(game.zx_diagram, game.zx_match_diagram)
        # The game already computed data and index — steal them to avoid recomputation
        state._data = game.data
        state._data_index = game.data_index
        state._diagram_stats = game.diagram_stats
        return state

    def clone(self) -> GameState:
        """
        Create an independent copy of this state for tree branching.

        The ZXDiagram is copied via its .copy() method.
        The ZXMatchDiagram is cloned by copying the underlying nx.DiGraph data structures
        directly (bypassing ZXMatchDiagram.__init__ and its overridden add_node/add_edge,
        which would recompute attributes and double-add to type sets). This replaces the
        previous approach of calling to_zx_match_diagram() from scratch, which was the
        dominant bottleneck in MCTS search (~5-15s per search of 100 simulations).

        Safety:
        - MatchNode/SuperNode objects are immutable (used as dict keys) — shared safely.
        - Node/edge attribute dicts are shallow-copied, so mutations don't propagate.
        - The type-specific node sets (frz_nodes, etc.) are shallow-copied sets.
        - The zx_diagram reference is updated so future add_node/add_edge calls
          (during efficient_rewrite in apply_action) read from the new diagram.
        """
        diagram_copy = self.zx_diagram.copy()
        match_diagram_copy = _clone_match_diagram(self.zx_match_diagram, diagram_copy)
        cloned = GameState(diagram_copy, match_diagram_copy)
        return cloned

    def ensure_data(self) -> tuple[Data, DataIndexToMatch]:
        """Lazily compute the PyG Data representation and DataIndexToMatch mapping."""
        if self._data is None:
            self._data, self._data_index = self.zx_match_diagram.to_pyg_data(True)
        return self._data, self._data_index

    @property
    def data(self) -> Data:
        self.ensure_data()
        return self._data

    @property
    def data_index(self) -> DataIndexToMatch:
        self.ensure_data()
        return self._data_index

    @property
    def diagram_stats(self) -> DiagramStats:
        if self._diagram_stats is None:
            self._diagram_stats = DiagramStats(self.zx_match_diagram)
        return self._diagram_stats

    def apply_action(self, action: tuple) -> tuple[float, bool]:
        """
        Apply an action to this state IN PLACE.

        The caller is responsible for calling clone() before this if the original
        state needs to be preserved (which it always does in MCTS).

        :param action: Action tuple as produced by AlphaZXDistribution.sample():
                       (graph_id, action_type, node_index, phase, new_edges, *transfer_edges)
        :return: (step_reward, is_terminal)
        """
        old_stats = self.diagram_stats

        # Decode and apply the rewrite
        data, data_index = self.ensure_data()
        match, params = tuple_to_match(self.zx_match_diagram, data, action, data_index)
        efficient_rewrite(self.zx_diagram, self.zx_match_diagram, match, params)

        # Cleanup: remove isolated nodes, self-loops, disconnected components
        removed_isolated = remove_isolated_nodes(self.zx_diagram)
        update_match_diagram_for_removed_nodes(self.zx_match_diagram, removed_isolated)
        remove_self_loop_edges(self.zx_diagram)
        removed_components = remove_isolated_components(self.zx_diagram)
        update_match_diagram_for_removed_nodes(self.zx_match_diagram, removed_components)

        # Invalidate cached data — will be lazily recomputed
        self._data = None
        self._data_index = None
        self._diagram_stats = None
        self._num_non_clifford = None
        self._cached_preprocessed_data = None

        # Compute reward
        new_stats = self.diagram_stats
        reward, _ = calculate_reward(old_stats, new_stats)

        # Check termination
        done = is_simplified(self.zx_diagram)

        return reward, done

    def is_terminal(self) -> bool:
        """Check if the diagram is fully simplified (no non-Clifford gates remain)."""
        return is_simplified(self.zx_diagram)

    def has_legal_actions(self) -> bool:
        """Check if there are any match nodes (legal rewrites) available.

        Uses the match node type sets on the ZXMatchDiagram for O(1) check
        instead of iterating all nodes.
        """
        for abbrev in METADATA.match_node_type_abbrevs:
            type_set = getattr(self.zx_match_diagram, f'{abbrev}_nodes', None)
            if type_set and len(type_set) > 0:
                return True
        return False

    @property
    def num_non_clifford(self) -> int:
        """Current number of non-Clifford (T) gates (cached)."""
        if self._num_non_clifford is None:
            self._num_non_clifford = num_non_clifford_gates(self.zx_diagram)
        return self._num_non_clifford

    @property
    def num_nodes(self) -> int:
        return self.zx_diagram.number_of_nodes()

    @property
    def num_edges(self) -> int:
        return self.zx_diagram.num_edges()

    def __repr__(self) -> str:
        return (
            f"GameState(nodes={self.num_nodes}, edges={self.num_edges}, "
            f"t_gates={self.num_non_clifford}, terminal={self.is_terminal()})"
        )
