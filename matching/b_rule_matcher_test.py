import unittest
import networkx as nx

from matching.b_rule_matcher import b_right_matches, b_right_pattern, b_left_matches, b_left_pattern
from matching.match import BLeftMatch, BRightMatch
from matching.zx_diagram import ZXDiagram


def diamond_graph() -> ZXDiagram:
    """
              b16---z4----x6---b17
                        X
    b18---z0---x2---z5----x7----z12---x14---b19
             X                      X
    b20---z1---x3---z8---x10----z13---x15---b21
                       X
              b22---z9---x11---b23
    """
    diagram = nx.compose_all(
        [b_left_pattern(), b_left_pattern(4, 5, 6, 7), b_left_pattern(8, 9, 10, 11), b_left_pattern(12, 13, 14, 15)])
    b16, b17, b18, b19, b20, b21, b22, b23 = diagram.add_b_nodes(8)
    diagram.add_s_edges_from(
        [(2, 5), (7, 12), (3, 8), (10, 13), (b16, 4), (6, b17), (b18, 0), (14, b19), (b20, 1), (15, b21), (b22, 9),
         (11, b23)])
    return diagram


def square_graph() -> ZXDiagram:
    """
    b4----z0---z1---b6
          |    |
    b5---x2---x3---b7
    """
    diagram = ZXDiagram()
    z0, z1 = diagram.add_z_nodes([0, 0])
    x2, x3 = diagram.add_x_nodes([0, 0])
    b4, b5, b6, b7 = diagram.add_b_nodes(4)
    diagram.add_s_edges_from([(b4, z0), (b5, x2), (z0, x2), (z0, z1), (x2, x3), (x3, z1), (z1, b6), (x3, b7)])
    return diagram


def square_graph_alternating() -> ZXDiagram:
    """
    b4----x0---z2---b6
          |    |
    b5---z1---x3---b7
    """
    diagram = ZXDiagram()
    x0 = diagram.add_x_node(0)
    z1, z2 = diagram.add_z_nodes([0, 0])
    x3 = diagram.add_x_node(0)
    b4, b5, b6, b7 = diagram.add_b_nodes(4)
    diagram.add_s_edges_from([(b4, x0), (b5, z1), (x0, z2), (x0, z1), (z2, x3), (x3, z1), (z2, b6), (x3, b7)])
    return diagram


class BRuleLeftTest(unittest.TestCase):

    def test_self_match(self):
        """
        b4---z0---x2---b6
                X
        b5---z1---x3---b7
        """
        diagram = b_left_pattern()
        b4, b5, b6, b7 = diagram.add_b_nodes(4)
        diagram.add_s_edges_from([(b4, 0), (b5, 1), (2, b6), (3, b7)])
        self.assertListEqual(list(b_left_matches(diagram)), [BLeftMatch(0, 2, 1, 3)])

    def test_self_match_connected_left_right(self):
        """
         z0---x2
        (   X   )
         z1---x3
        """
        diagram = b_left_pattern()
        diagram.add_s_edges_from([(0, 1), (2, 3)])
        self.assertListEqual(list(b_left_matches(diagram)), [BLeftMatch(0, 2, 1, 3)])

    def test_self_match_connected_top_bottom(self):
        """
          ------
        (       )
         z0---x2
            X
         z1---x3
        (       )
         -------
        """
        diagram = b_left_pattern()
        diagram.add_s_edges_from([(0, 2), (1, 3)])
        self.assertListEqual(list(b_left_matches(diagram)), [BLeftMatch(0, 2, 1, 3)])

    def test_self_match_connected_right_end(self):
        """
         b4---z0---x2
                 X   )
         b5---z1---x3
        """
        diagram = b_left_pattern()
        b4, b5 = diagram.add_b_nodes(2)
        diagram.add_s_edges_from([(b4, 0), (b5, 1), (2, 3)])
        self.assertListEqual(list(b_left_matches(diagram)), [BLeftMatch(0, 2, 1, 3)])

    def test_self_match_connected_left_end(self):
        """
          z0---x2---b4
         (   X
          z1---x3---b5
        """
        diagram = b_left_pattern()
        b4, b5 = diagram.add_b_nodes(2)
        diagram.add_s_edges_from([(0, 1), (2, b4), (3, b5)])
        self.assertListEqual(list(b_left_matches(diagram)), [BLeftMatch(0, 2, 1, 3)])

    def test_missing_x_no_match(self):
        """
        b0---z2---x4---b6

        b1---z3---x5---b7
        """
        diagram = ZXDiagram()
        b0, b1 = diagram.add_b_nodes(2)
        z2, z3 = diagram.add_z_nodes([0, 0])
        x4, x5 = diagram.add_x_nodes([0, 0])
        b6, b7 = diagram.add_b_nodes(2)
        diagram.add_s_edges_from([(b0, z2), (z2, x4), (x4, b6), (b1, z3), (z3, x5), (x5, b7)])
        self.assertListEqual(list(b_left_matches(diagram)), [])

    def test_two_way_parallel_composition(self):
        """
        b8----z0---x2---b10
                 X
        b9----z1---x3---b11
        b12---z4---x6---b14
                 X
        b13---z5---x7---b15
        """
        left_diagram = b_left_pattern()
        right_diagram = b_left_pattern(4, 5, 6, 7)
        diagram = nx.compose(left_diagram, right_diagram)
        b8, b9, b10, b11, b12, b13, b14, b15 = diagram.add_b_nodes(8)
        diagram.add_s_edges_from([(b8, 0), (b9, 1), (2, b10), (3, b11)])
        diagram.add_s_edges_from([(b12, 4), (b13, 5), (6, b14), (7, b15)])
        self.assertListEqual(list(b_left_matches(diagram)),
                             [BLeftMatch(1, 2, 0, 3), BLeftMatch(4, 6, 5, 7)])

    def test_two_way_parallel_composition_connected(self):
        """
        b8----z0---x2---b10
                 X
        b9----z1---x3---b11
                 X
        b12---z4---x6---b14
                 X
        b13---z5---x7---b15
        """
        diagram = nx.compose(b_left_pattern(), b_left_pattern(4, 5, 6, 7))
        diagram.add_s_edges_from([(1, 6), (4, 3)])
        b8, b9, b10, b11, b12, b13, b14, b15 = diagram.add_b_nodes(8)
        diagram.add_s_edges_from([(b8, 0), (b9, 1), (2, b10), (3, b11), (b12, 4), (b13, 5), (6, b14), (7, b15)])
        self.assertListEqual(list(b_left_matches(diagram)), [])

    def test_sequential_composition_zx(self):
        """
        b8---z0---x2---z4---x6---b10
                X         X
        b9---z1---x3---z5---x7---b11
        """
        diagram = nx.compose(b_left_pattern(), b_left_pattern(4, 5, 6, 7))
        diagram.add_s_edges_from([(2, 4), (3, 5)])
        b8, b9, b10, b11 = diagram.add_b_nodes(4)
        diagram.add_s_edges_from([(b8, 0), (b9, 1), (6, b10), (7, b11)])
        self.assertListEqual(list(b_left_matches(diagram)), [BLeftMatch(1, 2, 0, 3), BLeftMatch(4, 6, 5, 7)])

    def test_sequential_composition_zx_connected_ends(self):
        """
          --z0---x2---z4---x6--
        (      X         X      )
          --z1---x3---z5---x7--
        """
        diagram = nx.compose(b_left_pattern(), b_left_pattern(4, 5, 6, 7))
        diagram.add_s_edges_from([(2, 4), (3, 5), (0, 1), (6, 7)])
        self.assertListEqual(list(b_left_matches(diagram)), [BLeftMatch(1, 2, 0, 3), BLeftMatch(4, 6, 5, 7)])

    def test_sequential_composition_xx(self):
        """
        b8---z0---x2---x4---z6---b10
                X         X
        b9---z1---x3---x5---z7---b11
        """
        diagram = nx.compose(b_left_pattern(), b_left_pattern(7, 6, 5, 4))
        b8, b9, b10, b11 = diagram.add_b_nodes(4)
        diagram.add_s_edges_from([(2, 4), (3, 5), (b8, 0), (b9, 1), (6, b10), (7, b11)])
        # We don't care about the order of the node matches, just the nodes that were matched.
        self.assertListEqual(list(b_left_matches(diagram)), [BLeftMatch(7, 4, 6, 5), BLeftMatch(1, 2, 0, 3)])

    def test_sequential_composition_zz(self):
        """
        b8---x7---z5---z0---x2---b10
                X         X
        b9---x6---z4---z1---x3---b11
        """
        diagram = nx.compose(b_left_pattern(), b_left_pattern(4, 5, 6, 7))
        b8, b9, b10, b11 = diagram.add_b_nodes(4)
        diagram.add_s_edges_from([(4, 1), (5, 0), (b8, 7), (b9, 6), (2, b10), (3, b11)])
        self.assertListEqual(list(b_left_matches(diagram)), [BLeftMatch(1, 2, 0, 3), BLeftMatch(4, 6, 5, 7)])

    def test_upper_left_sequential_composition_zx(self):
        """
                  b8----z0---x2---b9
                           X
        b10---z4---x6---z1---x3---b11
                 X
        b12---z5---x7---b13
        """
        diagram = nx.compose(b_left_pattern(), b_left_pattern(4, 5, 6, 7))
        b8, b9, b10, b11, b12, b13 = diagram.add_b_nodes(6)
        diagram.add_s_edges_from([(b8, 0), (2, b9), (b10, 4), (6, 1), (3, b11), (b12, 5), (7, b13)])
        self.assertListEqual(list(b_left_matches(diagram)),
                             [BLeftMatch(1, 2, 0, 3), BLeftMatch(4, 6, 5, 7)])

    def test_upper_left_sequential_composition_zz(self):
        """
                   b8---z0---x2---b9
                           X
        b10---x7---z5---z1---x3---b11
                 X
        b12---x6---z4---b13
        """
        diagram = nx.compose(b_left_pattern(), b_left_pattern(4, 5, 6, 7))
        b8, b9, b10, b11, b12, b13 = diagram.add_b_nodes(6)
        diagram.add_s_edges_from([(b8, 0), (2, b9), (b10, 7), (5, 1), (3, b11), (b12, 6), (4, b13)])
        self.assertListEqual(list(b_left_matches(diagram)),
                             [BLeftMatch(1, 2, 0, 3), BLeftMatch(4, 6, 5, 7)])

    def test_upper_left_sequential_composition_xx(self):
        """
                   b8---x3---z1----b9
                           X
        b10---z4---x6---x2---z0---b11
                 X
        b12---z5---x7---b13
        """
        diagram = nx.compose(b_left_pattern(), b_left_pattern(4, 5, 6, 7))
        b8, b9, b10, b11, b12, b13 = diagram.add_b_nodes(6)
        diagram.add_s_edges_from([(6, 2), (b8, 3), (1, b9), (b10, 4), (0, b11), (b12, 5), (7, b13)])
        self.assertListEqual(list(b_left_matches(diagram)),
                             [BLeftMatch(1, 2, 0, 3), BLeftMatch(4, 6, 5, 7)])

    def test_upper_left_sequential_composition_xx_back_connected(self):
        """
                   b8----x3---z1---b9
                            X
         b10---z4---x6---x2---z0---b11
                  X
         b12---z5---x7---b13
        """
        diagram = nx.compose(b_left_pattern(), b_left_pattern(4, 5, 6, 7))
        b8, b9, b10, b11, b12, b13 = diagram.add_b_nodes(6)
        diagram.add_s_edges_from([(b8, 3), (1, b9), (b10, 4), (6, 2), (0, b11), (b12, 5), (7, b13)])
        self.assertListEqual(list(b_left_matches(diagram)),
                             [BLeftMatch(1, 2, 0, 3), BLeftMatch(4, 6, 5, 7)])

    def test_lower_right_sequential_composition_zx(self):
        """
        b8----z4---x6---b9
                 X
        b10---z5---x7---z0---x2---b11
                           X
                  b12---z1---x3---b13
        """
        diagram = nx.compose(b_left_pattern(), b_left_pattern(4, 5, 6, 7))
        b8, b9, b10, b11, b12, b13 = diagram.add_b_nodes(6)
        diagram.add_s_edges_from([(b8, 4), (6, b9), (b10, 5), (7, 0), (2, b11), (b12, 1), (3, b13)])
        self.assertListEqual(list(b_left_matches(diagram)),
                             [BLeftMatch(1, 2, 0, 3), BLeftMatch(4, 6, 5, 7)])

    def test_lower_right_sequential_composition_zz(self):
        """
        b8----x7---z5---b9
                 X
        b10---x6---z4---z0---x2---b11
                           X
                  b12---z1---x3---b13
        """
        diagram = nx.compose(b_left_pattern(), b_left_pattern(4, 5, 6, 7))
        b8, b9, b10, b11, b12, b13 = diagram.add_b_nodes(6)
        diagram.add_s_edges_from([(b8, 7), (5, b9), (b10, 6), (4, 0), (2, b11), (b12, 1), (3, b13)])
        self.assertListEqual(list(b_left_matches(diagram)),
                             [BLeftMatch(1, 2, 0, 3), BLeftMatch(4, 6, 5, 7)])

    def test_lower_right_sequential_composition_xx(self):
        """
        b8----z4---x6---b9
                 X
        b10---z5---x7---x3---z1---b11
                           X
                  b12---x2---z0---b13
        """
        diagram = nx.compose(b_left_pattern(), b_left_pattern(4, 5, 6, 7))
        b8, b9, b10, b11, b12, b13 = diagram.add_b_nodes(6)
        diagram.add_s_edges_from([(b8, 4), (6, b9), (b10, 5), (7, 3), (1, b11), (b12, 2), (0, b13)])
        self.assertListEqual(list(b_left_matches(diagram)),
                             [BLeftMatch(1, 2, 0, 3), BLeftMatch(4, 6, 5, 7)])

    def test_five_way_cross_composition(self):
        """
         b16----z0---x2---b17  b18---z4----x6---b19
                   X                     X
         b20----z1---x3--------------z5----x7---b21
                             X
         b22----z8--x10--------------z12--x14---b23
                   X                     X
         b24----z9--x11--b25   b26---z13--x15---b27
        """
        diagram = nx.compose_all([b_left_pattern(), b_left_pattern(4, 5, 6, 7), b_left_pattern(8, 9, 10, 11),
                                  b_left_pattern(12, 13, 14, 15)])
        b16, b17, b18, b19, b20, b21, b22, b23, b24, b25, b26, b27 = diagram.add_b_nodes(12)
        diagram.add_s_edges_from(
            [(b16, 0), (2, b17), (b18, 4), (6, b19), (b20, 1), (7, b21), (b22, 8), (14, b23), (b24, 9), (11, b25),
             (b26, 13), (15, b27), (3, 12), (10, 5)])
        self.assertListEqual(list(b_left_matches(diagram)),
                             [BLeftMatch(9, 10, 8, 11), BLeftMatch(13, 14, 12, 15), BLeftMatch(1, 2, 0, 3),
                              BLeftMatch(4, 6, 5, 7)])

    def test_five_way_cross_composition_connected(self):
        """
         ---z0----x2--------------z8----x10---
        (      X                     X        )
         ---z1----x3----z4---x6---z9----x11---
                           X
         ---z12---x14---z5---x7---z16---x18---
        (       X                     X       )
         ---z13---x15-------------z17---x19---
        """
        diagram = nx.compose_all([b_left_pattern(), b_left_pattern(4, 5, 6, 7), b_left_pattern(8, 9, 10, 11),
                                  b_left_pattern(12, 13, 14, 15), b_left_pattern(16, 17, 18, 19)])
        diagram.add_edges_from(
            [(2, 8), (0, 1), (10, 11), (3, 4), (6, 9), (14, 5), (7, 16), (12, 13), (18, 19), (15, 17)])
        self.assertListEqual(list(b_left_matches(diagram)),
                             [BLeftMatch(4, 6, 5, 7), BLeftMatch(0, 2, 1, 3), BLeftMatch(9, 11, 8, 10),
                              BLeftMatch(17, 18, 16, 19), BLeftMatch(13, 14, 12, 15)])

    def test_diamond_composition(self):
        """
                  b16---z4----x6---b17
                            X
        b18---z0---x2---z5----x7----z12---x14---b19
                 X                      X
        b20---z1---x3---z8---x10----z13---x15---b21
                           X
                  b22---z9---x11---b23
        """
        self.assertListEqual(list(b_left_matches(diamond_graph())),
                             [BLeftMatch(4, 6, 5, 7), BLeftMatch(0, 2, 1, 3), BLeftMatch(9, 11, 8, 10),
                              BLeftMatch(13, 14, 12, 15)])

    def test_square(self):
        """
        b4----z0---z1---b6
              |    |
        b5---x2---x3---b7
        """
        self.assertListEqual(list(b_left_matches(square_graph())), [])

    def test_square_alternating(self):
        """
        b4----x0---z2---b6
              |    |
        b5---z1---x3---b7
        """
        self.assertListEqual(list(b_left_matches(square_graph_alternating())), [BLeftMatch(1, 0, 2, 3)])


def line_graph() -> ZXDiagram:
    diagram = ZXDiagram()
    diagram.add_x_node(0)
    diagram.add_z_node(0)
    b2, b3 = diagram.add_b_nodes(2)
    diagram.add_s_edges_from([(b2, 0), (0, 1), (1, b3)])
    return diagram


def nonzero_phase_no_match_test_graph() -> ZXDiagram:
    diagram = ZXDiagram()
    diagram.add_z_node(0)
    diagram.add_x_node(1.5)
    b2, b3, b4, b5 = diagram.add_b_nodes(4)
    diagram.add_s_edges_from([(b2, 0), (b3, 0), (1, b4), (1, b5)])
    return diagram


def two_identity_test_graph() -> ZXDiagram:
    diagram = b_right_pattern()
    diagram.add_s_edges_from([(0, 0), (1, 1)])
    return diagram


class BRuleRightTest(unittest.TestCase):

    def test_self_match_z(self):
        diagram = b_right_pattern()
        b2, b3, b4, b5 = diagram.add_b_nodes(4)
        diagram.add_s_edges_from([(b2, 0), (b3, 0), (1, b4), (1, b5)])
        self.assertListEqual(list(b_right_matches(diagram)), [BRightMatch(0, 1)])

    def test_line_graph_no_match(self):
        self.assertListEqual(list(b_right_matches(line_graph())), [])

    def test_nonzero_phase_no_match_z(self):
        self.assertListEqual(list(b_right_matches(nonzero_phase_no_match_test_graph())), [])

    def test_two_identity_match(self):
        self.assertListEqual(list(b_right_matches(two_identity_test_graph())), [BRightMatch(0, 1)])
