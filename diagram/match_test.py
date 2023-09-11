import unittest

import networkx as nx

from diagram.match import Basis, FLeftZMatch, FLeftXMatch, FRightZMatch, FRightXMatch, BLeftMatch, BRightMatch, \
    YLeftZMatch, YLeftXMatch, YRightZMatch, YRightXMatch
from diagram.match_patterns import f_left_z_pattern, f_left_x_pattern, f_right_z_pattern, f_right_x_pattern, \
    basis_to_ntype_index, y_left_z_pattern, y_left_x_pattern, y_left_pattern, y_right_z_pattern, y_right_x_pattern, \
    b_left_pattern, b_right_pattern
from diagram.pyzx_graph_generator import nx_clifford_graph
from diagram.pyzx_nx_conv import S_ETYPE_INDEX, H_ETYPE_INDEX, Z_NTYPE_NAME, X_NTYPE_NAME
from diagram.zx_diagram import ZXDiagram
from diagram.zx_match_diagram import ZXMatchDiagram


def consecutive_parallel_edge_test_graph(basis: Basis) -> ZXDiagram:
    node_type = basis_to_ntype_index(basis)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(23, type=node_type)
    nx_graph.add_node(45, type=node_type)
    nx_graph.add_node(57, type=node_type)
    nx_graph.add_edge(23, 45, type=S_ETYPE_INDEX)
    nx_graph.add_edge(23, 45, type=S_ETYPE_INDEX)
    nx_graph.add_edge(23, 45, type=S_ETYPE_INDEX)
    nx_graph.add_edge(45, 57, type=S_ETYPE_INDEX)
    nx_graph.add_edge(45, 57, type=S_ETYPE_INDEX)
    return ZXDiagram(nx_graph)


def hadamard_edge_test_graph(basis: Basis) -> ZXDiagram:
    node_type = basis_to_ntype_index(basis)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_type)
    nx_graph.add_node(1, type=node_type)
    nx_graph.add_edge(0, 1, type=H_ETYPE_INDEX)
    return ZXDiagram(nx_graph)


def simple_hadamard_edge_test_graph(basis: Basis) -> ZXDiagram:
    node_type = basis_to_ntype_index(basis)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_type)
    nx_graph.add_node(1, type=node_type)
    nx_graph.add_edge(0, 1, type=S_ETYPE_INDEX)
    nx_graph.add_edge(0, 1, type=H_ETYPE_INDEX)
    return ZXDiagram(nx_graph)


def disconnected_test_graph(basis: Basis) -> ZXDiagram:
    node_type = basis_to_ntype_index(basis)
    nx_graph = nx.MultiGraph()
    nx_graph.add_node(0, type=node_type)
    nx_graph.add_node(1, type=node_type)
    return ZXDiagram(nx_graph)


class FLeftMatchTest(unittest.TestCase):

    def test_self_match_z(self):
        self.assertListEqual(list(f_left_z_pattern().f_left_z_matches()), [FLeftZMatch(0, 1)])

    def test_self_match_x(self):
        self.assertListEqual(list(f_left_x_pattern().f_left_x_matches()), [FLeftXMatch(0, 1)])

    def test_parallel_edge_match_z(self):
        self.assertListEqual(list(consecutive_parallel_edge_test_graph(Z_NTYPE_NAME).f_left_z_matches()),
                             [FLeftZMatch(23, 45), FLeftZMatch(45, 57)])

    def test_parallel_edge_match_x(self):
        self.assertListEqual(list(consecutive_parallel_edge_test_graph(X_NTYPE_NAME).f_left_x_matches()),
                             [FLeftXMatch(23, 45), FLeftXMatch(45, 57)])

    def test_hadamard_edge_no_match_z(self):
        self.assertListEqual(list(hadamard_edge_test_graph(Z_NTYPE_NAME).f_left_z_matches()), [])

    def test_hadamard_edge_no_match_x(self):
        self.assertListEqual(list(hadamard_edge_test_graph(X_NTYPE_NAME).f_left_x_matches()), [])

    def test_hadamard_edge_match_z(self):
        self.assertListEqual(list(simple_hadamard_edge_test_graph(Z_NTYPE_NAME).f_left_z_matches()), [])

    def test_hadamard_edge_match_x(self):
        self.assertListEqual(list(simple_hadamard_edge_test_graph(X_NTYPE_NAME).f_left_x_matches()), [])

    def test_disconnected_no_match_z(self):
        self.assertListEqual(list(disconnected_test_graph(Z_NTYPE_NAME).f_left_z_matches()), [])

    def test_disconnected_no_match_x(self):
        self.assertListEqual(list(disconnected_test_graph(X_NTYPE_NAME).f_left_x_matches()), [])


class FRightMatchTest(unittest.TestCase):

    def test_self_match_z(self):
        self.assertListEqual(list(f_right_z_pattern().f_right_z_matches()), [FRightZMatch(0)])

    def test_self_match_x(self):
        self.assertListEqual(list(f_right_x_pattern().f_right_x_matches()), [FRightXMatch(0)])

    def test_match_f_left_pattern_z(self):
        self.assertListEqual(list(f_left_z_pattern().f_right_z_matches()), [FRightZMatch(0), FRightZMatch(1)])

    def test_match_f_left_pattern_x(self):
        self.assertListEqual(list(f_left_x_pattern().f_right_x_matches()), [FRightXMatch(0), FRightXMatch(1)])

    def test_different_basis_no_match_z(self):
        self.assertListEqual(list(f_right_x_pattern().f_right_z_matches()), [])

    def test_different_basis_no_match_x(self):
        self.assertListEqual(list(f_right_z_pattern().f_right_x_matches()), [])

    def test_disconnected_match_z(self):
        self.assertListEqual(list(disconnected_test_graph(Z_NTYPE_NAME).f_right_z_matches()),
                             [FRightZMatch(0), FRightZMatch(1)])

    def test_disconnected_match_x(self):
        self.assertListEqual(list(disconnected_test_graph(X_NTYPE_NAME).f_right_x_matches()),
                             [FRightXMatch(0), FRightXMatch(1)])

    def test_consecutive_parallel_edge_match_z(self):
        self.assertListEqual(list(consecutive_parallel_edge_test_graph(Z_NTYPE_NAME).f_right_z_matches()),
                             [FRightZMatch(57), FRightZMatch(45), FRightZMatch(23)])

    def test_consecutive_parallel_edge_match_x(self):
        self.assertListEqual(list(consecutive_parallel_edge_test_graph(X_NTYPE_NAME).f_right_x_matches()),
                             [FRightXMatch(57), FRightXMatch(45), FRightXMatch(23)])


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
        diagram = b_left_pattern(15, 4, 2, 8)
        b4, b5, b6, b7 = diagram.add_b_nodes(4)
        diagram.add_s_edges_from([(b4, 15), (b5, 4), (2, b6), (8, b7)])
        self.assertListEqual(list(diagram.b_left_matches()), [BLeftMatch(4, 8, 15, 2)])

    def test_self_match_connected_left_right(self):
        """
         z0---x2
        (   X   )
         z1---x3
        """
        diagram = b_left_pattern()
        diagram.add_s_edges_from([(0, 1), (2, 3)])
        self.assertListEqual(list(diagram.b_left_matches()), [BLeftMatch(0, 2, 1, 3)])

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
        self.assertListEqual(list(diagram.b_left_matches()), [BLeftMatch(0, 2, 1, 3)])

    def test_self_match_connected_right_end(self):
        """
         b4---z0---x2
                 X   )
         b5---z1---x3
        """
        diagram = b_left_pattern()
        b4, b5 = diagram.add_b_nodes(2)
        diagram.add_s_edges_from([(b4, 0), (b5, 1), (2, 3)])
        self.assertListEqual(list(diagram.b_left_matches()), [BLeftMatch(0, 2, 1, 3)])

    def test_self_match_connected_left_end(self):
        """
          z0---x2---b4
         (   X
          z1---x3---b5
        """
        diagram = b_left_pattern()
        b4, b5 = diagram.add_b_nodes(2)
        diagram.add_s_edges_from([(0, 1), (2, b4), (3, b5)])
        self.assertListEqual(list(diagram.b_left_matches()), [BLeftMatch(0, 2, 1, 3)])

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
        self.assertListEqual(list(diagram.b_left_matches()), [])

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
        self.assertListEqual(list(diagram.b_left_matches()),
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
        self.assertListEqual(list(diagram.b_left_matches()), [])

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
        self.assertListEqual(list(diagram.b_left_matches()), [BLeftMatch(1, 2, 0, 3), BLeftMatch(4, 6, 5, 7)])

    def test_sequential_composition_zx_connected_ends(self):
        """
          --z0---x2---z4---x6--
        (      X         X      )
          --z1---x3---z5---x7--
        """
        diagram = nx.compose(b_left_pattern(), b_left_pattern(4, 5, 6, 7))
        diagram.add_s_edges_from([(2, 4), (3, 5), (0, 1), (6, 7)])
        self.assertListEqual(list(diagram.b_left_matches()), [BLeftMatch(1, 2, 0, 3), BLeftMatch(4, 6, 5, 7)])

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
        self.assertListEqual(list(diagram.b_left_matches()), [BLeftMatch(7, 4, 6, 5), BLeftMatch(1, 2, 0, 3)])

    def test_sequential_composition_zz(self):
        """
        b8---x7---z5---z0---x2---b10
                X         X
        b9---x6---z4---z1---x3---b11
        """
        diagram = nx.compose(b_left_pattern(), b_left_pattern(4, 5, 6, 7))
        b8, b9, b10, b11 = diagram.add_b_nodes(4)
        diagram.add_s_edges_from([(4, 1), (5, 0), (b8, 7), (b9, 6), (2, b10), (3, b11)])
        self.assertListEqual(list(diagram.b_left_matches()), [BLeftMatch(1, 2, 0, 3), BLeftMatch(4, 6, 5, 7)])

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
        self.assertListEqual(list(diagram.b_left_matches()),
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
        self.assertListEqual(list(diagram.b_left_matches()),
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
        self.assertListEqual(list(diagram.b_left_matches()),
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
        self.assertListEqual(list(diagram.b_left_matches()),
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
        self.assertListEqual(list(diagram.b_left_matches()),
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
        self.assertListEqual(list(diagram.b_left_matches()),
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
        self.assertListEqual(list(diagram.b_left_matches()),
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
        self.assertListEqual(list(diagram.b_left_matches()),
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
        self.assertListEqual(list(diagram.b_left_matches()),
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
        self.assertListEqual(list(diamond_graph().b_left_matches()),
                             [BLeftMatch(4, 6, 5, 7), BLeftMatch(0, 2, 1, 3), BLeftMatch(9, 11, 8, 10),
                              BLeftMatch(13, 14, 12, 15)])

    def test_square(self):
        """
        b4----z0---z1---b6
              |    |
        b5---x2---x3---b7
        """
        self.assertListEqual(list(square_graph().b_left_matches()), [])

    def test_square_alternating(self):
        """
        b4----x0---z2---b6
              |    |
        b5---z1---x3---b7
        """
        self.assertListEqual(list(square_graph_alternating().b_left_matches()), [BLeftMatch(1, 0, 2, 3)])


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
        self.assertListEqual(list(diagram.b_right_matches()), [BRightMatch(0, 1)])

    def test_line_graph_no_match(self):
        self.assertListEqual(list(line_graph().b_right_matches()), [])

    def test_nonzero_phase_no_match_z(self):
        self.assertListEqual(list(nonzero_phase_no_match_test_graph().b_right_matches()), [])

    def test_two_identity_match(self):
        self.assertListEqual(list(two_identity_test_graph().b_right_matches()), [BRightMatch(0, 1)])


def parallel_edge_no_match_test_graph(basis: Basis, first: bool = False, second: bool = False,
                                      third=False) -> ZXDiagram:
    diagram = y_left_pattern(basis)
    b4, b5, b6 = diagram.add_b_nodes(3)
    diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
    if first:
        diagram.add_s_edge(0, 1)
    if second:
        diagram.add_s_edge(1, 2)
    if third:
        diagram.add_s_edge(1, 3)
    return diagram


def disconnected_no_match_test_graph(basis: Basis, first: bool = False, second: bool = False,
                                     third=False) -> ZXDiagram:
    diagram = y_left_pattern(basis)
    b4, b5, b6 = diagram.add_b_nodes(3)
    diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
    if first:
        diagram.remove_edge(0, 1)
    if second:
        diagram.remove_edge(1, 2)
    if third:
        diagram.remove_edge(1, 3)
    return diagram


class YLeftMatchTest(unittest.TestCase):

    def test_self_match_z(self):
        diagram = y_left_z_pattern()
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
        self.assertListEqual(list(diagram.y_left_z_matches()), [YLeftZMatch(0, 1, 2, 3)])

    def test_self_match_x(self):
        diagram = y_left_x_pattern()
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
        self.assertListEqual(list(diagram.y_left_x_matches()), [YLeftXMatch(0, 1, 2, 3)])

    def test_parallel_edge_no_match_z(self):
        for a in [True, False]:
            for b in [True, False]:
                for c in [True, False]:
                    matches = list(parallel_edge_no_match_test_graph(Z_NTYPE_NAME, a, b, c).y_left_z_matches())
                    if not (a or b or c):
                        self.assertListEqual(matches, [YLeftZMatch(0, 1, 2, 3)])
                    else:
                        self.assertListEqual(matches, [])

    def test_parallel_edge_no_match_x(self):
        for a in [True, False]:
            for b in [True, False]:
                for c in [True, False]:
                    matches = list(parallel_edge_no_match_test_graph(X_NTYPE_NAME, a, b, c).y_left_x_matches())
                    if not (a or b or c):
                        self.assertListEqual(matches, [YLeftXMatch(0, 1, 2, 3)])
                    else:
                        self.assertListEqual(matches, [])

    def test_disconnected_no_match_z(self):
        for a in [True, False]:
            for b in [True, False]:
                for c in [True, False]:
                    matches = list(disconnected_no_match_test_graph(Z_NTYPE_NAME, a, b, c).y_left_z_matches())
                    if not (a or b or c):
                        self.assertListEqual(matches, [YLeftZMatch(0, 1, 2, 3)])
                    else:
                        self.assertListEqual(matches, [])

    def test_disconnected_no_match_x(self):
        for a in [True, False]:
            for b in [True, False]:
                for c in [True, False]:
                    matches = list(disconnected_no_match_test_graph(X_NTYPE_NAME, a, b, c).y_left_x_matches())
                    if not (a or b or c):
                        self.assertListEqual(matches, [YLeftXMatch(0, 1, 2, 3)])
                    else:
                        self.assertListEqual(matches, [])


class YRightMatchTest(unittest.TestCase):

    def test_self_match_z(self):
        diagram = y_right_z_pattern()
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
        self.assertListEqual(list(diagram.y_right_z_matches()), [YRightZMatch(0, 1, 2, 3)])

    def test_self_match_x(self):
        diagram = y_right_x_pattern()
        b4, b5, b6 = diagram.add_b_nodes(3)
        diagram.add_s_edges_from([(b4, 0), (2, b5), (3, b6)])
        self.assertListEqual(list(diagram.y_right_x_matches()), [YRightXMatch(0, 1, 2, 3)])

