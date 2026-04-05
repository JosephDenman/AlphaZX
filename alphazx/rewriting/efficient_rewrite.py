"""
Efficient incremental rewriting for ZX diagrams.

This module provides functions to update a ZXMatchDiagram incrementally after a rewrite,
avoiding the O(n²) cost of recomputing all matches from scratch.

The key insight is that a rewrite only affects matches within a bounded neighborhood
of the rewrite location. We use a 4-hop neighborhood for safety, even though 2-hop
is theoretically sufficient.
"""

from typing import Set, Iterator

from alphazx.diagram.match import (
    MatchNode, FRightZMatch, FRightXMatch, FLeftZMatch, FLeftXMatch,
    BRightMatch, BLeftMatch, YRightZMatch, YRightXMatch, YLeftZMatch, YLeftXMatch,
    BoundaryMatch, SimpleMatchNode, CompoundMatchNode, SuperNode
)
from alphazx.diagram.zx_diagram import ZXDiagram, base_match_from_node
from alphazx.diagram.zx_match_diagram import (
    ZXMatchDiagram, add_match
)
from alphazx.rewriting.update_set import UpdateSet
from alphazx.rewriting.utils import rewrite


def get_k_hop_neighborhood(zx_diagram: ZXDiagram, nodes: Set[int], k: int) -> Set[int]:
    """
    Get all nodes within k hops of the given nodes.

    :param zx_diagram: The ZX diagram
    :param nodes: Starting nodes
    :param k: Number of hops
    :return: Set of all nodes within k hops (including the starting nodes)
    """
    neighborhood = set(nodes)
    frontier = set(nodes)

    for _ in range(k):
        new_frontier = set()
        for node in frontier:
            if zx_diagram.has_node(node):
                for neighbor in zx_diagram.neighbors(node):
                    if neighbor not in neighborhood:
                        new_frontier.add(neighbor)
                        neighborhood.add(neighbor)
        frontier = new_frontier
        if not frontier:
            break

    return neighborhood


def get_matches_involving_nodes(zx_match_diagram: ZXMatchDiagram, nodes: Set[int]) -> Set[MatchNode]:
    """
    Get all matches in the match diagram that involve any of the given ZX diagram nodes.

    Uses set intersection for O(min(|match|, |nodes|)) per match instead of
    O(|match| * |nodes|) from the naive linear scan.

    :param zx_match_diagram: The match diagram
    :param nodes: Set of ZX diagram node IDs
    :return: Set of matches involving those nodes
    """
    matches = set()
    for match_node in zx_match_diagram.nodes():
        if isinstance(match_node, MatchNode):
            # _nodes is a tuple; checking membership against the `nodes` set
            # is O(1) per element.  For small matches (1-4 nodes) this is fast.
            # We avoid calling the .nodes property which allocates a new list.
            if not nodes.isdisjoint(match_node._nodes):
                matches.add(match_node)
    return matches


# ============================================================================
# Neighborhood-based match detection functions
# ============================================================================

def detect_boundary_matches_in_neighborhood(
    zx_diagram: ZXDiagram,
    neighborhood: Set[int]
) -> Iterator[BoundaryMatch]:
    """Detect BoundaryMatch instances for boundary nodes in the neighborhood."""
    for n in neighborhood:
        if zx_diagram.has_node(n) and zx_diagram.is_boundary(n):
            yield BoundaryMatch(n)


def detect_f_right_z_matches_in_neighborhood(
    zx_diagram: ZXDiagram,
    neighborhood: Set[int]
) -> Iterator[FRightZMatch]:
    """Detect FRightZMatch instances for Z-basis nodes in the neighborhood."""
    for n in neighborhood:
        if zx_diagram.has_node(n) and zx_diagram.is_z_basis(n):
            yield FRightZMatch(n)


def detect_f_right_x_matches_in_neighborhood(
    zx_diagram: ZXDiagram,
    neighborhood: Set[int]
) -> Iterator[FRightXMatch]:
    """Detect FRightXMatch instances for X-basis nodes in the neighborhood."""
    for n in neighborhood:
        if zx_diagram.has_node(n) and zx_diagram.is_x_basis(n):
            yield FRightXMatch(n)


def detect_f_left_z_matches_in_neighborhood(
    zx_diagram: ZXDiagram,
    neighborhood: Set[int]
) -> Iterator[FLeftZMatch]:
    """Detect FLeftZMatch instances (adjacent Z-Z pairs) where at least one node is in neighborhood."""
    seen = set()
    for n in neighborhood:
        if not zx_diagram.has_node(n) or not zx_diagram.is_z_basis(n):
            continue
        for neighbor in zx_diagram.neighbors(n):
            if zx_diagram.is_z_basis(neighbor):
                # Create canonical pair (smaller, larger) to avoid duplicates
                pair = (min(n, neighbor), max(n, neighbor))
                if pair not in seen:
                    seen.add(pair)
                    yield FLeftZMatch(pair[0], pair[1])


def detect_f_left_x_matches_in_neighborhood(
    zx_diagram: ZXDiagram,
    neighborhood: Set[int]
) -> Iterator[FLeftXMatch]:
    """Detect FLeftXMatch instances (adjacent X-X pairs) where at least one node is in neighborhood."""
    seen = set()
    for n in neighborhood:
        if not zx_diagram.has_node(n) or not zx_diagram.is_x_basis(n):
            continue
        for neighbor in zx_diagram.neighbors(n):
            if zx_diagram.is_x_basis(neighbor):
                pair = (min(n, neighbor), max(n, neighbor))
                if pair not in seen:
                    seen.add(pair)
                    yield FLeftXMatch(pair[0], pair[1])


def detect_b_right_matches_in_neighborhood(
    zx_diagram: ZXDiagram,
    neighborhood: Set[int]
) -> Iterator[BRightMatch]:
    """
    Detect BRightMatch instances (adjacent Z-X pairs, both degree 3, both phase 0)
    where at least one node is in neighborhood.
    """
    seen = set()
    # Iterate over edges, not just nodes
    for n in neighborhood:
        if not zx_diagram.has_node(n):
            continue
        if not zx_diagram.is_basis(n):
            continue
        if zx_diagram.degree(n) != 3:
            continue
        if zx_diagram.phase(n) != 0:
            continue

        for neighbor in zx_diagram.neighbors(n):
            if not zx_diagram.is_basis(neighbor):
                continue
            # Types must be different
            if zx_diagram.type(n) == zx_diagram.type(neighbor):
                continue
            if zx_diagram.degree(neighbor) != 3:
                continue
            if zx_diagram.phase(neighbor) != 0:
                continue

            # Found a valid pair - order as (Z, X)
            if zx_diagram.is_z_basis(n) and zx_diagram.is_x_basis(neighbor):
                z_node, x_node = n, neighbor
            elif zx_diagram.is_z_basis(neighbor) and zx_diagram.is_x_basis(n):
                z_node, x_node = neighbor, n
            else:
                continue

            pair = (z_node, x_node)
            if pair not in seen:
                seen.add(pair)
                yield BRightMatch(z_node, x_node)


def detect_b_left_matches_in_neighborhood(
    zx_diagram: ZXDiagram,
    neighborhood: Set[int]
) -> Iterator[BLeftMatch]:
    """
    Detect BLeftMatch instances (z-x-m-n cycle pattern)
    where at least one node is in neighborhood.

    BLeftMatch requires:
    - 4 nodes: z, x, m, n
    - z, m are Z-basis; x, n are X-basis
    - All have degree 3 and phase 0
    - Forms cycle: z-x, x-m, m-n, n-z
    """
    seen = set()

    # Start from Z nodes in neighborhood
    for z in neighborhood:
        if not zx_diagram.has_node(z):
            continue
        if not zx_diagram.is_z_basis(z):
            continue
        if zx_diagram.degree(z) != 3 or zx_diagram.phase(z) != 0:
            continue

        # Find X neighbors of z
        for x in zx_diagram.neighbors(z):
            if not zx_diagram.is_x_basis(x):
                continue
            if zx_diagram.degree(x) != 3 or zx_diagram.phase(x) != 0:
                continue

            # Find Z neighbors of x (other than z)
            for m in zx_diagram.neighbors(x):
                if m == z:
                    continue
                if not zx_diagram.is_z_basis(m):
                    continue
                if zx_diagram.degree(m) != 3 or zx_diagram.phase(m) != 0:
                    continue

                # Find X neighbors of m (other than x)
                for n in zx_diagram.neighbors(m):
                    if n == x:
                        continue
                    if not zx_diagram.is_x_basis(n):
                        continue
                    if zx_diagram.degree(n) != 3 or zx_diagram.phase(n) != 0:
                        continue

                    # Check if n connects back to z (completing the cycle)
                    if z in zx_diagram.neighbors(n):
                        # Found a valid BLeftMatch
                        # Use canonical form to avoid duplicates
                        match_tuple = tuple(sorted([z, x, m, n]))
                        if match_tuple not in seen:
                            seen.add(match_tuple)
                            yield BLeftMatch(z, x, m, n)


def detect_y_right_z_matches_in_neighborhood(
    zx_diagram: ZXDiagram,
    neighborhood: Set[int]
) -> Iterator[YRightZMatch]:
    """
    Detect YRightZMatch instances where at least one node is in neighborhood.

    YRightZMatch requires:
    - Center node is X-basis with degree 3 and phase 1.5 (= -0.5 mod 2)
    - All 3 neighbors are Z-basis with degree 2
    - Sum of neighbor phases mod 2 = 1.5 (= -0.5 mod 2)
    """
    from alphazx.diagram.zx_diagram import _phase_eq, _y_right_sort_key

    for center in neighborhood:
        if not zx_diagram.has_node(center):
            continue
        if not zx_diagram.is_x_basis(center):
            continue
        if zx_diagram.degree(center) != 3:
            continue
        if not _phase_eq(zx_diagram.phase(center), 1.5):
            continue

        neighbors = list(zx_diagram.neighbors(center))
        if len(neighbors) != 3:
            continue

        # Check all neighbors are Z-basis with degree 2
        if not all(zx_diagram.is_z_basis(n) and zx_diagram.degree(n) == 2 for n in neighbors):
            continue

        # Check phase sum (mod 2 to handle normalized phases)
        phase_sum = sum(zx_diagram.phase(n) for n in neighbors) % 2
        if not _phase_eq(phase_sum, 1.5):
            continue

        # Sort: 0.5 node first, then the two 1.5 nodes
        sorted_neighbors = sorted(neighbors, key=lambda n: _y_right_sort_key(zx_diagram.phase(n)))
        yield YRightZMatch(sorted_neighbors[0], center, sorted_neighbors[1], sorted_neighbors[2])


def detect_y_right_x_matches_in_neighborhood(
    zx_diagram: ZXDiagram,
    neighborhood: Set[int]
) -> Iterator[YRightXMatch]:
    """
    Detect YRightXMatch instances where at least one node is in neighborhood.

    YRightXMatch requires:
    - Center node is Z-basis with degree 3 and phase 1.5 (= -0.5 mod 2)
    - All 3 neighbors are X-basis with degree 2
    - Sum of neighbor phases mod 2 = 1.5 (= -0.5 mod 2)
    """
    from alphazx.diagram.zx_diagram import _phase_eq, _y_right_sort_key

    for center in neighborhood:
        if not zx_diagram.has_node(center):
            continue
        if not zx_diagram.is_z_basis(center):
            continue
        if zx_diagram.degree(center) != 3:
            continue
        if not _phase_eq(zx_diagram.phase(center), 1.5):
            continue

        neighbors = list(zx_diagram.neighbors(center))
        if len(neighbors) != 3:
            continue

        if not all(zx_diagram.is_x_basis(n) and zx_diagram.degree(n) == 2 for n in neighbors):
            continue

        phase_sum = sum(zx_diagram.phase(n) for n in neighbors) % 2
        if not _phase_eq(phase_sum, 1.5):
            continue

        sorted_neighbors = sorted(neighbors, key=lambda n: _y_right_sort_key(zx_diagram.phase(n)))
        yield YRightXMatch(sorted_neighbors[0], center, sorted_neighbors[1], sorted_neighbors[2])


def detect_y_left_z_matches_in_neighborhood(
    zx_diagram: ZXDiagram,
    neighborhood: Set[int]
) -> Iterator[YLeftZMatch]:
    """
    Detect YLeftZMatch instances where at least one node is in neighborhood.

    YLeftZMatch requires:
    - Center node is X-basis with degree 3 and phase 0
    - All 3 neighbors are Z-basis with degree 2
    - Sum of neighbor phases mod 2 = 0.5
    """
    from alphazx.diagram.zx_diagram import _phase_eq, _y_left_sort_key

    for center in neighborhood:
        if not zx_diagram.has_node(center):
            continue
        if not zx_diagram.is_x_basis(center):
            continue
        if zx_diagram.degree(center) != 3:
            continue
        if zx_diagram.phase(center) != 0:
            continue

        neighbors = list(zx_diagram.neighbors(center))
        if len(neighbors) != 3:
            continue

        if not all(zx_diagram.is_z_basis(n) and zx_diagram.degree(n) == 2 for n in neighbors):
            continue

        phase_sum = sum(zx_diagram.phase(n) for n in neighbors) % 2
        if not _phase_eq(phase_sum, 0.5):
            continue

        # Sort: 1.5 node first (= -0.5 before normalization), then the two 0.5 nodes
        sorted_neighbors = sorted(neighbors, key=lambda n: _y_left_sort_key(zx_diagram.phase(n)))
        yield YLeftZMatch(sorted_neighbors[0], center, sorted_neighbors[1], sorted_neighbors[2])


def detect_y_left_x_matches_in_neighborhood(
    zx_diagram: ZXDiagram,
    neighborhood: Set[int]
) -> Iterator[YLeftXMatch]:
    """
    Detect YLeftXMatch instances where at least one node is in neighborhood.

    YLeftXMatch requires:
    - Center node is Z-basis with degree 3 and phase 0
    - All 3 neighbors are X-basis with degree 2
    - Sum of neighbor phases mod 2 = 0.5
    """
    from alphazx.diagram.zx_diagram import _phase_eq, _y_left_sort_key

    for center in neighborhood:
        if not zx_diagram.has_node(center):
            continue
        if not zx_diagram.is_z_basis(center):
            continue
        if zx_diagram.degree(center) != 3:
            continue
        if zx_diagram.phase(center) != 0:
            continue

        neighbors = list(zx_diagram.neighbors(center))
        if len(neighbors) != 3:
            continue

        if not all(zx_diagram.is_x_basis(n) and zx_diagram.degree(n) == 2 for n in neighbors):
            continue

        phase_sum = sum(zx_diagram.phase(n) for n in neighbors) % 2
        if not _phase_eq(phase_sum, 0.5):
            continue

        sorted_neighbors = sorted(neighbors, key=lambda n: _y_left_sort_key(zx_diagram.phase(n)))
        yield YLeftXMatch(sorted_neighbors[0], center, sorted_neighbors[1], sorted_neighbors[2])


def detect_all_matches_in_neighborhood(
    zx_diagram: ZXDiagram,
    neighborhood: Set[int]
) -> Set[MatchNode]:
    """
    Detect all matches where at least one node is in the neighborhood.

    :param zx_diagram: The ZX diagram
    :param neighborhood: Set of ZX diagram node IDs to search
    :return: Set of all matches involving nodes in the neighborhood
    """
    matches = set()

    # Boundary matches
    for match in detect_boundary_matches_in_neighborhood(zx_diagram, neighborhood):
        matches.add(match)

    # F-Right matches (single nodes)
    for match in detect_f_right_z_matches_in_neighborhood(zx_diagram, neighborhood):
        matches.add(match)
    for match in detect_f_right_x_matches_in_neighborhood(zx_diagram, neighborhood):
        matches.add(match)

    # F-Left matches (adjacent same-basis pairs)
    for match in detect_f_left_z_matches_in_neighborhood(zx_diagram, neighborhood):
        matches.add(match)
    for match in detect_f_left_x_matches_in_neighborhood(zx_diagram, neighborhood):
        matches.add(match)

    # B-Right matches (adjacent Z-X pairs)
    for match in detect_b_right_matches_in_neighborhood(zx_diagram, neighborhood):
        matches.add(match)

    # B-Left matches (z-x-m-n cycles)
    for match in detect_b_left_matches_in_neighborhood(zx_diagram, neighborhood):
        matches.add(match)

    # Y-Right matches
    for match in detect_y_right_z_matches_in_neighborhood(zx_diagram, neighborhood):
        matches.add(match)
    for match in detect_y_right_x_matches_in_neighborhood(zx_diagram, neighborhood):
        matches.add(match)

    # Y-Left matches
    for match in detect_y_left_z_matches_in_neighborhood(zx_diagram, neighborhood):
        matches.add(match)
    for match in detect_y_left_x_matches_in_neighborhood(zx_diagram, neighborhood):
        matches.add(match)

    return matches


# ============================================================================
# Match diagram update functions
# ============================================================================

def remove_match_from_diagram(zx_match_diagram: ZXMatchDiagram, match: MatchNode) -> None:
    """
    Remove a match node from the match diagram, including its edges.

    Note: This does NOT remove the super node, as it may be shared by other matches.
    """
    if not zx_match_diagram.has_node(match):
        return

    # Remove from the type-specific set
    type_set = getattr(zx_match_diagram, f'{match.abbrev}_nodes', None)
    if type_set is not None and match in type_set:
        type_set.discard(match)

    # Remove the node (this also removes all incident edges)
    zx_match_diagram.remove_node(match)


def add_match_to_diagram_safe(
    zx_match_diagram: ZXMatchDiagram,
    zx_diagram: ZXDiagram,
    match: MatchNode
) -> bool:
    """
    Add a match node to the match diagram with proper edges.
    Returns True if successfully added, False if the match references non-existent nodes.

    :param zx_match_diagram: The match diagram to update
    :param zx_diagram: The ZX diagram (for computing attributes)
    :param match: The match to add
    :return: True if successfully added
    """
    if zx_match_diagram.has_node(match):
        return True  # Already exists

    # Verify all nodes in the match still exist
    for node in match.nodes:
        if not zx_diagram.has_node(node):
            return False

    # Use the existing add_match function from zx_match_diagram module
    try:
        add_match(zx_match_diagram, zx_diagram, match)
        return True
    except Exception:
        return False


# ============================================================================
# Main efficient rewrite function
# ============================================================================

def efficient_rewrite(
    zx_diagram: ZXDiagram,
    zx_match_diagram: ZXMatchDiagram,
    match: MatchNode,
    f_right_params: tuple[float, int, set[int]] | None = None,
    neighborhood_hops: int = 4
) -> UpdateSet:
    """
    Performs the given rewrite and incrementally updates the match diagram.

    This avoids recomputing all matches by only checking within a bounded neighborhood
    of the rewrite location.

    :param zx_diagram: The ZX diagram (will be mutated)
    :param zx_match_diagram: The match diagram (will be mutated)
    :param match: The match to apply
    :param f_right_params: Parameters for F-right rewrites (phase, new_edges, transfer_edges)
    :param neighborhood_hops: Number of hops to check (default 4 for safety)
    :return: UpdateSet containing added/removed nodes
    """
    # Step 1: Get the nodes affected by the rewrite
    affected_nodes = set(match.nodes)

    # Step 2: Compute the k-hop neighborhood BEFORE the rewrite
    pre_rewrite_neighborhood = get_k_hop_neighborhood(zx_diagram, affected_nodes, neighborhood_hops)

    # Step 3: Get all matches currently in the neighborhood
    old_matches = get_matches_involving_nodes(zx_match_diagram, pre_rewrite_neighborhood)

    # Step 4: Perform the actual rewrite on the ZX diagram
    update_set = rewrite(zx_diagram, match, f_right_params)

    # Step 5: Compute the neighborhood AFTER the rewrite (includes new nodes)
    post_rewrite_neighborhood = pre_rewrite_neighborhood.copy()
    post_rewrite_neighborhood.update(update_set.added_nodes)

    # Extend neighborhood from new nodes
    for new_node in update_set.added_nodes:
        extended = get_k_hop_neighborhood(zx_diagram, {new_node}, neighborhood_hops)
        post_rewrite_neighborhood.update(extended)

    # Remove nodes that no longer exist
    post_rewrite_neighborhood = {n for n in post_rewrite_neighborhood if zx_diagram.has_node(n)}

    # Step 6: Remove old matches from the match diagram
    for old_match in old_matches:
        remove_match_from_diagram(zx_match_diagram, old_match)

    # Step 7: Detect new matches in the post-rewrite neighborhood
    new_matches = detect_all_matches_in_neighborhood(zx_diagram, post_rewrite_neighborhood)

    # Step 8: Add new matches to the match diagram
    for new_match in new_matches:
        add_match_to_diagram_safe(zx_match_diagram, zx_diagram, new_match)

    return update_set


def verify_match_diagram_consistency(
    zx_diagram: ZXDiagram,
    zx_match_diagram: ZXMatchDiagram
) -> tuple[bool, list[str]]:
    """
    Verify that the match diagram is consistent with a fresh computation.

    :param zx_diagram: The ZX diagram
    :param zx_match_diagram: The match diagram to verify
    :return: Tuple of (is_consistent, list of error messages)
    """
    from alphazx.diagram.zx_match_diagram import to_zx_match_diagram

    errors = []

    # Compute fresh match diagram
    fresh_diagram = to_zx_match_diagram(zx_diagram)

    # Get all matches from both
    incremental_matches = {m for m in zx_match_diagram.nodes() if isinstance(m, MatchNode)}
    fresh_matches = {m for m in fresh_diagram.nodes() if isinstance(m, MatchNode)}

    # Check for missing matches
    missing = fresh_matches - incremental_matches
    if missing:
        errors.append(f"Missing matches in incremental diagram: {missing}")

    # Check for extra matches
    extra = incremental_matches - fresh_matches
    if extra:
        errors.append(f"Extra matches in incremental diagram: {extra}")

    return len(errors) == 0, errors
