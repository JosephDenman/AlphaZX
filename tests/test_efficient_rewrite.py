"""
Unit tests for efficient_rewrite comparing incremental match diagram updates
against full recomputation.
"""
import unittest
from copy import deepcopy

from alphazx.diagram.diagram_generators import clifford_zx_diagram
from alphazx.diagram.match import (
    FRightZMatch, FRightXMatch, FLeftZMatch, FLeftXMatch,
    BRightMatch, BLeftMatch, YRightZMatch, YRightXMatch, YLeftZMatch, YLeftXMatch,
    MatchNode
)
from alphazx.diagram.zx_diagram import ZXDiagram
from alphazx.diagram.zx_match_diagram import to_zx_match_diagram, ZXMatchDiagram
from alphazx.rewriting.efficient_rewrite import (
    efficient_rewrite,
    verify_match_diagram_consistency,
    get_k_hop_neighborhood,
    detect_all_matches_in_neighborhood
)
from alphazx.rewriting.utils import rewrite
from match_patterns import (
    b_right_pattern, b_left_pattern,
    y_left_z_pattern, y_left_x_pattern,
    y_right_z_pattern, y_right_x_pattern,
    f_left_z_pattern, f_left_x_pattern,
    f_right_z_pattern, f_right_x_pattern
)


# Phase denominator for all diagrams
PD = 4


def get_match_set(zx_match_diagram: ZXMatchDiagram) -> set[MatchNode]:
    """Extract all match nodes from a match diagram."""
    return {m for m in zx_match_diagram.nodes() if isinstance(m, MatchNode)}


def compare_match_diagrams(incremental: ZXMatchDiagram, fresh: ZXMatchDiagram) -> tuple[set, set]:
    """
    Compare two match diagrams and return sets of missing and extra matches.

    Returns:
        Tuple of (missing_in_incremental, extra_in_incremental)
    """
    inc_matches = get_match_set(incremental)
    fresh_matches = get_match_set(fresh)

    missing = fresh_matches - inc_matches
    extra = inc_matches - fresh_matches

    return missing, extra


class TestGetKHopNeighborhood(unittest.TestCase):
    """Test the neighborhood computation function."""

    def test_1_hop_neighborhood(self):
        """Test that 1-hop gets direct neighbors."""
        diagram = ZXDiagram(PD)
        n0 = diagram.add_z_node(0)
        n1 = diagram.add_z_node(0)
        n2 = diagram.add_z_node(0)
        n3 = diagram.add_z_node(0)
        diagram.add_s_edge(n0, n1)
        diagram.add_s_edge(n1, n2)
        diagram.add_s_edge(n2, n3)

        neighborhood = get_k_hop_neighborhood(diagram, {n0}, 1)
        self.assertEqual(neighborhood, {n0, n1})

    def test_2_hop_neighborhood(self):
        """Test that 2-hop gets neighbors of neighbors."""
        diagram = ZXDiagram(PD)
        n0 = diagram.add_z_node(0)
        n1 = diagram.add_z_node(0)
        n2 = diagram.add_z_node(0)
        n3 = diagram.add_z_node(0)
        diagram.add_s_edge(n0, n1)
        diagram.add_s_edge(n1, n2)
        diagram.add_s_edge(n2, n3)

        neighborhood = get_k_hop_neighborhood(diagram, {n0}, 2)
        self.assertEqual(neighborhood, {n0, n1, n2})

    def test_4_hop_neighborhood(self):
        """Test 4-hop neighborhood on a chain."""
        diagram = ZXDiagram(PD)
        nodes = [diagram.add_z_node(0) for _ in range(6)]
        for i in range(5):
            diagram.add_s_edge(nodes[i], nodes[i+1])

        neighborhood = get_k_hop_neighborhood(diagram, {nodes[0]}, 4)
        self.assertEqual(neighborhood, set(nodes[:5]))


class TestDetectMatchesInNeighborhood(unittest.TestCase):
    """Test the match detection functions."""

    def test_detect_f_right_matches(self):
        """Test detection of F-right matches."""
        diagram = ZXDiagram(PD)
        n0 = diagram.add_z_node(0.25)
        n1 = diagram.add_x_node(0.5)

        matches = detect_all_matches_in_neighborhood(diagram, {n0, n1})

        self.assertIn(FRightZMatch(n0), matches)
        self.assertIn(FRightXMatch(n1), matches)

    def test_detect_f_left_matches(self):
        """Test detection of F-left matches (adjacent same-basis pairs)."""
        diagram = f_left_z_pattern(PD)

        matches = detect_all_matches_in_neighborhood(diagram, {0, 1})

        self.assertIn(FLeftZMatch(0, 1), matches)
        self.assertIn(FRightZMatch(0), matches)
        self.assertIn(FRightZMatch(1), matches)

    def test_detect_b_right_matches(self):
        """Test detection of B-right matches.

        B-right requires both nodes to have degree 3 and phase 0.
        We need to add extra connections to get degree 3.
        """
        diagram = b_right_pattern(PD)
        # Add boundary nodes to get degree 3 for both z and x
        b0, b1, b2, b3 = diagram.add_b_nodes(4)
        diagram.add_s_edge(b0, 0)  # z gets second edge
        diagram.add_s_edge(b1, 0)  # z gets third edge (degree 3)
        diagram.add_s_edge(1, b2)  # x gets second edge
        diagram.add_s_edge(1, b3)  # x gets third edge (degree 3)

        matches = detect_all_matches_in_neighborhood(diagram, {0, 1})

        b_right_found = any(isinstance(m, BRightMatch) for m in matches)
        self.assertTrue(b_right_found)

    def test_detect_b_left_matches(self):
        """Test detection of B-left matches.

        B-left requires all 4 nodes to have degree 3 and phase 0.
        b_left_pattern creates the cycle, we need to add extra edges.
        """
        diagram = b_left_pattern(PD)
        # Add boundary nodes to get degree 3 for all nodes (they already have degree 2 from the cycle)
        b4, b5, b6, b7 = diagram.add_b_nodes(4)
        diagram.add_s_edge(b4, 0)  # z0 gets third edge
        diagram.add_s_edge(b5, 1)  # z1 gets third edge
        diagram.add_s_edge(2, b6)  # x2 gets third edge
        diagram.add_s_edge(3, b7)  # x3 gets third edge

        matches = detect_all_matches_in_neighborhood(diagram, {0, 1, 2, 3})

        b_left_found = any(isinstance(m, BLeftMatch) for m in matches)
        self.assertTrue(b_left_found)


class TestEfficientRewriteFLeft(unittest.TestCase):
    """Test efficient_rewrite for F-left rewrites."""

    def test_f_left_z_rewrite(self):
        """Test F-left Z rewrite produces consistent match diagram."""
        diagram = f_left_z_pattern(PD)
        # Add some extra nodes to make it more realistic
        b0, b1 = diagram.add_b_nodes(2)
        diagram.add_s_edge(b0, 0)
        diagram.add_s_edge(1, b1)

        zx_match_diagram = to_zx_match_diagram(diagram)
        f_left_matches = list(diagram.f_left_z_matches())
        self.assertTrue(len(f_left_matches) > 0)

        match = f_left_matches[0]
        efficient_rewrite(diagram, zx_match_diagram, match)

        is_consistent, errors = verify_match_diagram_consistency(diagram, zx_match_diagram)
        self.assertTrue(is_consistent, f"Inconsistency found: {errors}")

    def test_f_left_x_rewrite(self):
        """Test F-left X rewrite produces consistent match diagram."""
        diagram = f_left_x_pattern(PD)
        b0, b1 = diagram.add_b_nodes(2)
        diagram.add_s_edge(b0, 0)
        diagram.add_s_edge(1, b1)

        zx_match_diagram = to_zx_match_diagram(diagram)
        f_left_matches = list(diagram.f_left_x_matches())
        self.assertTrue(len(f_left_matches) > 0)

        match = f_left_matches[0]
        efficient_rewrite(diagram, zx_match_diagram, match)

        is_consistent, errors = verify_match_diagram_consistency(diagram, zx_match_diagram)
        self.assertTrue(is_consistent, f"Inconsistency found: {errors}")


class TestEfficientRewriteBRight(unittest.TestCase):
    """Test efficient_rewrite for B-right rewrites."""

    def test_b_right_rewrite(self):
        """Test B-right rewrite produces consistent match diagram."""
        diagram = b_right_pattern(PD)
        # Add boundary nodes
        b0, b1, b2, b3 = diagram.add_b_nodes(4)
        diagram.add_s_edge(b0, 0)
        diagram.add_s_edge(b1, 0)
        diagram.add_s_edge(1, b2)
        diagram.add_s_edge(1, b3)

        zx_match_diagram = to_zx_match_diagram(diagram)
        b_right_matches = list(diagram.b_right_matches())
        self.assertTrue(len(b_right_matches) > 0)

        match = b_right_matches[0]
        efficient_rewrite(diagram, zx_match_diagram, match)

        is_consistent, errors = verify_match_diagram_consistency(diagram, zx_match_diagram)
        self.assertTrue(is_consistent, f"Inconsistency found: {errors}")


class TestEfficientRewriteBLeft(unittest.TestCase):
    """Test efficient_rewrite for B-left rewrites."""

    def test_b_left_rewrite(self):
        """Test B-left rewrite produces consistent match diagram."""
        diagram = b_left_pattern(PD)
        b4, b5, b6, b7 = diagram.add_b_nodes(4)
        diagram.add_s_edges_from([(b4, 0), (b5, 1), (2, b6), (3, b7)])

        zx_match_diagram = to_zx_match_diagram(diagram)
        b_left_matches = list(diagram.b_left_matches())
        self.assertTrue(len(b_left_matches) > 0)

        match = b_left_matches[0]
        efficient_rewrite(diagram, zx_match_diagram, match)

        is_consistent, errors = verify_match_diagram_consistency(diagram, zx_match_diagram)
        self.assertTrue(is_consistent, f"Inconsistency found: {errors}")


class TestEfficientRewriteYRules(unittest.TestCase):
    """Test efficient_rewrite for Y rewrites."""

    def test_y_left_z_rewrite(self):
        """Test Y-left Z rewrite produces consistent match diagram."""
        diagram = y_left_z_pattern(PD)
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])

        zx_match_diagram = to_zx_match_diagram(diagram)
        y_left_matches = list(diagram.y_left_matches())
        self.assertTrue(len(y_left_matches) > 0)

        match = y_left_matches[0]
        efficient_rewrite(diagram, zx_match_diagram, match)

        is_consistent, errors = verify_match_diagram_consistency(diagram, zx_match_diagram)
        self.assertTrue(is_consistent, f"Inconsistency found: {errors}")

    def test_y_left_x_rewrite(self):
        """Test Y-left X rewrite produces consistent match diagram."""
        diagram = y_left_x_pattern(PD)
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])

        zx_match_diagram = to_zx_match_diagram(diagram)
        y_left_matches = list(diagram.y_left_matches())
        self.assertTrue(len(y_left_matches) > 0)

        match = y_left_matches[0]
        efficient_rewrite(diagram, zx_match_diagram, match)

        is_consistent, errors = verify_match_diagram_consistency(diagram, zx_match_diagram)
        self.assertTrue(is_consistent, f"Inconsistency found: {errors}")

    def test_y_right_z_rewrite(self):
        """Test Y-right Z rewrite produces consistent match diagram."""
        diagram = y_right_z_pattern(PD)
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])

        zx_match_diagram = to_zx_match_diagram(diagram)
        y_right_matches = list(diagram.y_right_matches())
        self.assertTrue(len(y_right_matches) > 0)

        match = y_right_matches[0]
        efficient_rewrite(diagram, zx_match_diagram, match)

        is_consistent, errors = verify_match_diagram_consistency(diagram, zx_match_diagram)
        self.assertTrue(is_consistent, f"Inconsistency found: {errors}")

    def test_y_right_x_rewrite(self):
        """Test Y-right X rewrite produces consistent match diagram."""
        diagram = y_right_x_pattern(PD)
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])

        zx_match_diagram = to_zx_match_diagram(diagram)
        y_right_matches = list(diagram.y_right_matches())
        self.assertTrue(len(y_right_matches) > 0)

        match = y_right_matches[0]
        efficient_rewrite(diagram, zx_match_diagram, match)

        is_consistent, errors = verify_match_diagram_consistency(diagram, zx_match_diagram)
        self.assertTrue(is_consistent, f"Inconsistency found: {errors}")


class TestEfficientRewriteMultipleRewrites(unittest.TestCase):
    """Test that multiple sequential rewrites remain consistent."""

    def test_multiple_rewrites_remain_consistent(self):
        """Test applying multiple rewrites in sequence."""
        # Create a more complex diagram
        diagram = clifford_zx_diagram(3, 5, t_gates=False)
        zx_match_diagram = to_zx_match_diagram(diagram)

        max_rewrites = 10
        rewrites_applied = 0

        for _ in range(max_rewrites):
            # Try to find any applicable match
            f_left_matches = list(diagram.f_left_matches())
            if f_left_matches:
                match = f_left_matches[0]
                efficient_rewrite(diagram, zx_match_diagram, match)
                rewrites_applied += 1

                # Verify consistency after each rewrite
                is_consistent, errors = verify_match_diagram_consistency(diagram, zx_match_diagram)
                self.assertTrue(is_consistent,
                    f"Inconsistency after rewrite {rewrites_applied}: {errors}")
                continue

            b_left_matches = list(diagram.b_left_matches())
            if b_left_matches:
                match = b_left_matches[0]
                efficient_rewrite(diagram, zx_match_diagram, match)
                rewrites_applied += 1

                is_consistent, errors = verify_match_diagram_consistency(diagram, zx_match_diagram)
                self.assertTrue(is_consistent,
                    f"Inconsistency after rewrite {rewrites_applied}: {errors}")
                continue

            # No more matches
            break

        # Should have applied at least some rewrites
        self.assertGreater(rewrites_applied, 0, "No rewrites were applied")


class TestEfficientRewriteVsFullRecomputation(unittest.TestCase):
    """Compare efficient_rewrite against full recomputation."""

    def test_incremental_matches_full_recomputation(self):
        """Verify incremental update produces same matches as full recomputation."""
        diagram = clifford_zx_diagram(2, 3, t_gates=False)

        # Create a copy for incremental update
        diagram_incremental = diagram.copy()
        # Create match diagram from the copy (so internal reference is correct)
        zx_match_diagram_incremental = to_zx_match_diagram(diagram_incremental)

        f_left_matches = list(diagram_incremental.f_left_matches())
        if f_left_matches:
            match = f_left_matches[0]
            efficient_rewrite(diagram_incremental, zx_match_diagram_incremental, match)

            # Apply same rewrite using full recomputation on another copy
            diagram_full = diagram.copy()
            rewrite(diagram_full, match)
            zx_match_diagram_full = to_zx_match_diagram(diagram_full)

            # Compare match sets
            missing, extra = compare_match_diagrams(
                zx_match_diagram_incremental,
                zx_match_diagram_full
            )

            self.assertEqual(missing, set(), f"Missing matches: {missing}")
            self.assertEqual(extra, set(), f"Extra matches: {extra}")


class TestVerifyMatchDiagramConsistency(unittest.TestCase):
    """Test the verification function itself."""

    def test_consistent_diagram_passes(self):
        """Test that a freshly created diagram is consistent with itself."""
        diagram = clifford_zx_diagram(2, 3, t_gates=False)
        zx_match_diagram = to_zx_match_diagram(diagram)

        is_consistent, errors = verify_match_diagram_consistency(diagram, zx_match_diagram)
        self.assertTrue(is_consistent, f"Fresh diagram should be consistent: {errors}")

    def test_modified_diagram_detected(self):
        """Test that inconsistencies are detected."""
        diagram = clifford_zx_diagram(2, 3, t_gates=False)
        zx_match_diagram = to_zx_match_diagram(diagram)

        # Modify the underlying diagram without updating match diagram
        f_left_matches = list(diagram.f_left_matches())
        if f_left_matches:
            match = f_left_matches[0]
            rewrite(diagram, match)  # This doesn't update zx_match_diagram

            is_consistent, errors = verify_match_diagram_consistency(diagram, zx_match_diagram)
            # Should detect inconsistency
            self.assertFalse(is_consistent, "Should detect inconsistency after untracked rewrite")


if __name__ == '__main__':
    unittest.main()
