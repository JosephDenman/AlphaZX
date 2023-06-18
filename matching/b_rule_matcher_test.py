import unittest
from typing import List, Dict, Tuple

import networkx as nx

from graph.pyzx_nx_conv import S_ETYPE_INDEX, Z_NTYPE_NAME, X_NTYPE_NAME, H_ETYPE_INDEX, COLUMN, ROW, \
    Z_NTYPE_INDEX, X_NTYPE_INDEX, DEGREE
from matching.b_rule_matcher import b_right_matches, b_right_pattern, b_left_matches, b_left_pattern
from matching.match_types import RuleMode, BLeftMatch, BRightMatch
from matching.utils import rule_mode_to_ntype_indices


def diamond_graph() -> nx.MultiGraph:
    """
              ---z4---x8---
                    X
    ---z0---x2---z5---x9----z12---x14---
          X                     X
    ---z1---x3---z6---x10---z13---x15---
                    X
              ---z7---x11---
    """
    bottom_graph = b_left_pattern()
    center_left_graph = b_left_pattern(4, 5, 8, 9)
    center_right_graph = b_left_pattern(6, 7, 10, 11)
    top_graph = b_left_pattern(12, 13, 14, 15)
    graph = nx.compose_all([bottom_graph, center_left_graph, center_right_graph, top_graph])
    graph.add_edges_from([(2, 5), (9, 12), (3, 6), (10, 13)],
                         type=S_ETYPE_INDEX)
    return graph


def square_graph(bottom_left: int = 0, bottom_right: int = 1, top_left: int = 2, top_right: int = 3) -> nx.MultiGraph:
    """
    ---z0---z1---
        |    |
    ---x1---x2---
    """
    nx_graph = nx.MultiGraph()
    nx_graph.add_nodes_from([bottom_left, top_left], type=Z_NTYPE_INDEX, phase=0, degree=3)
    nx_graph.add_nodes_from([bottom_right, top_right], type=X_NTYPE_INDEX, phase=0, degree=3)
    nx_graph.add_edges_from(
        [(bottom_left, bottom_right), (bottom_left, top_left), (bottom_right, top_right), (top_left, top_right)],
        type=S_ETYPE_INDEX)
    return nx_graph


def square_graph_alternating(bottom_left: int = 0, bottom_right: int = 1, top_left: int = 2,
                             top_right: int = 3) -> nx.MultiGraph:
    """
    ---x0---z2---
        |    |
    ---z1---x3---
    """
    nx_graph = nx.MultiGraph()
    nx_graph.add_nodes_from([bottom_left, top_right], type=X_NTYPE_INDEX, phase=0, degree=3)
    nx_graph.add_nodes_from([bottom_right, top_left], type=Z_NTYPE_INDEX, phase=0, degree=3)
    nx_graph.add_edges_from(
        [(bottom_left, bottom_right), (bottom_left, top_left), (bottom_right, top_right), (top_left, top_right)],
        type=S_ETYPE_INDEX)
    return nx_graph


class BRuleLeftTest(unittest.TestCase):

    def test_self_match(self):
        """
        ---z0---x2---
              X
        ---z1---x3---
        """
        self.assertListEqual(list(b_left_matches(b_left_pattern(0, 1, 2, 3))), [BLeftMatch(0, 1, 2, 3)])

    def test_self_match_connected_left_right(self):
        """
         z0---x2
        (   X   )
         z1---x3
        """
        graph = b_left_pattern(0, 1, 2, 3)
        graph.add_edges_from([(0, 1), (2, 3)], type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(b_left_pattern(0, 1, 2, 3))), [BLeftMatch(0, 1, 2, 3)])

    def test_self_match_connected_top_bottom(self):
        """
         -------
        (       )
         z0---x2
            X
         z1---x3
        (       )
         -------
        """
        graph = b_left_pattern(0, 1, 2, 3)
        graph.add_edges_from([(0, 2), (1, 3)], type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(b_left_pattern(0, 1, 2, 3))), [BLeftMatch(0, 1, 2, 3)])

    def test_self_match_connected_right_end(self):
        """
         ---z0---x2
               X   )
         ---z1---x3
        """
        graph = b_left_pattern(0, 1, 2, 3)
        graph.add_edges_from([(2, 3)], type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(b_left_pattern(0, 1, 2, 3))), [BLeftMatch(0, 1, 2, 3)])

    def test_self_match_connected_left_end(self):
        """
          z0---x2---
         (   X
          z1---x3---
        """
        graph = b_left_pattern(0, 1, 2, 3)
        graph.add_edges_from([(2, 3)], type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(b_left_pattern(0, 1, 2, 3))), [BLeftMatch(0, 1, 2, 3)])

    def test_missing_x_no_match(self):
        """
        ---z0---x2---

        ---z1---x3---
        """
        graph = nx.MultiGraph()
        graph.add_nodes_from([0, 1], phase=0, type=Z_NTYPE_INDEX, degree=2)
        graph.add_nodes_from([2, 3], phase=0, type=X_NTYPE_INDEX, degree=2)
        graph.add_edges_from([(0, 2), (1, 3)], type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(graph)), [])

    def test_two_way_parallel_composition(self):
        """
        ---z0---x2---
              X
        ---z1---x3---
        ---z4---x6---
              X
        ---z5---x7---
        """
        left_graph = b_left_pattern()
        right_graph = b_left_pattern(4, 5, 6, 7)
        graph = nx.compose(left_graph, right_graph)
        self.assertListEqual(list(b_left_matches(graph)),
                             [BLeftMatch(0, 1, 2, 3), BLeftMatch(4, 5, 6, 7)])

    def test_two_way_parallel_composition_connected(self):
        """
        ---z0---x2---
              X
        ---z1---x3---
              X
        ---z4---x6---
              X
        ---z5---x7---
        """
        left_graph = b_left_pattern()
        right_graph = b_left_pattern(4, 5, 6, 7)
        graph = nx.compose(left_graph, right_graph)
        graph.add_edges_from([(1, 6), (3, 4)], type=S_ETYPE_INDEX)
        graph.nodes[1][DEGREE] = 4
        graph.nodes[3][DEGREE] = 4
        graph.nodes[4][DEGREE] = 4
        graph.nodes[6][DEGREE] = 4
        self.assertListEqual(list(b_left_matches(graph)), [])

    def test_sequential_composition_zx(self):
        """
        ---z0---x2---z4---x6---
              X         X
        ---z1---x3---z5---x7---
        """
        bottom_graph = b_left_pattern()
        top_graph = b_left_pattern(4, 5, 6, 7)
        graph = nx.compose(bottom_graph, top_graph)
        graph.add_edges_from([(2, 4), (3, 5)], type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(graph)),
                             [BLeftMatch(0, 1, 2, 3), BLeftMatch(4, 5, 6, 7)])

    def test_sequential_composition_zx_connected_ends(self):
        """
         ---z0---x2---z4---x6---
        (      X         X      )
         ---z1---x3---z5---x7---
        """
        bottom_graph = b_left_pattern()
        top_graph = b_left_pattern(4, 5, 6, 7)
        graph = nx.compose(bottom_graph, top_graph)
        graph.add_edges_from([(2, 4), (3, 5), (0, 1), (6, 7)], type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(graph)),
                             [BLeftMatch(0, 1, 2, 3), BLeftMatch(4, 5, 6, 7)])

    def test_sequential_composition_xx(self):
        """
        ---z0---x2---x4---z6---
              X         X
        ---z1---x3---x5---z7---
        """
        bottom_graph = b_left_pattern()
        top_graph = b_left_pattern(7, 6, 5, 4)
        graph = nx.compose(bottom_graph, top_graph)
        graph.add_edges_from([(2, 4), (3, 5)], type=S_ETYPE_INDEX)
        # We don't care about the order of the node matches, just the nodes that were matched.
        self.assertListEqual(list(b_left_matches(graph)),
                             [BLeftMatch(0, 1, 2, 3), BLeftMatch(7, 6, 4, 5)])

    def test_sequential_composition_zz(self):
        """
        ---x7---z5---z0---x2---
              X         X
        ---x6---z4---z1---x3---
        """
        bottom_pattern = b_left_pattern()
        top_pattern = b_left_pattern(4, 5, 6, 7)
        graph = nx.compose(bottom_pattern, top_pattern)
        graph.add_edges_from([(4, 1), (5, 0)], type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(graph)),
                             [BLeftMatch(0, 1, 2, 3), BLeftMatch(4, 5, 6, 7)])

    def test_upper_left_sequential_composition_zx(self):
        """
                  ---z0---x2---
                        X
        ---z4---x6---z1---x3---
              X
        ---z5---x7---
        """
        bottom_pattern = b_left_pattern()
        top_pattern = b_left_pattern(4, 5, 6, 7)
        graph = nx.compose(bottom_pattern, top_pattern)
        graph.add_edges_from([(6, 1)], type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(graph)),
                             [BLeftMatch(0, 1, 2, 3), BLeftMatch(4, 5, 6, 7)])

    def test_upper_left_sequential_composition_zz(self):
        """
                  ---z0---x2---
                        X
        ---x7---z5---z1---x3---
              X
        ---x6---z4---
        """
        bottom_pattern = b_left_pattern()
        top_pattern = b_left_pattern(4, 5, 6, 7)
        graph = nx.compose(bottom_pattern, top_pattern)
        graph.add_edges_from([(5, 1)], type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(graph)),
                             [BLeftMatch(0, 1, 2, 3), BLeftMatch(4, 5, 6, 7)])

    def test_upper_left_sequential_composition_xx(self):
        """
                  ---x3---z1---
                        X
        ---z4---x6---x2---z0---
              X
        ---z5---x7---
        """
        bottom_pattern = b_left_pattern()
        top_pattern = b_left_pattern(4, 5, 6, 7)
        graph = nx.compose(bottom_pattern, top_pattern)
        graph.add_edges_from([(6, 2)], type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(graph)),
                             [BLeftMatch(0, 1, 2, 3), BLeftMatch(4, 5, 6, 7)])

    def test_upper_left_sequential_composition_xx_back_connected(self):
        """
                   ---x3---z1---
                         X
         ---z4---x6---x2---z0---
              X
         ---z5---x7---
        """
        top_graph = b_left_pattern()
        bottom_graph = b_left_pattern(4, 5, 6, 7)
        graph = nx.compose(top_graph, bottom_graph)
        graph.add_edges_from([(6, 2), (0, 7)], type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(graph)),
                             [BLeftMatch(0, 1, 2, 3), BLeftMatch(4, 5, 6, 7)])

    def test_lower_right_sequential_composition_zx(self):
        """
        ---z4---x6---
              X
        ---z5---x7---z0---x2---
                        X
                  ---z1---x3---
        """
        top_graph = b_left_pattern()
        bottom_graph = b_left_pattern(4, 5, 6, 7)
        graph = nx.compose(top_graph, bottom_graph)
        graph.add_edges_from([(7, 0)], type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(graph)),
                             [BLeftMatch(0, 1, 2, 3), BLeftMatch(4, 5, 6, 7)])

    def test_lower_right_sequential_composition_zz(self):
        """
        ---x7---z5---
              X
        ---x6---z4---z0---x2---
                        X
                  ---z1---x3---
        """
        top_graph = b_left_pattern()
        bottom_graph = b_left_pattern(4, 5, 6, 7)
        graph = nx.compose(top_graph, bottom_graph)
        graph.add_edges_from([(4, 0)], type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(graph)),
                             [BLeftMatch(0, 1, 2, 3), BLeftMatch(4, 5, 6, 7)])

    def test_lower_right_sequential_composition_xx(self):
        """
        ---z4---x6---
              X
        ---z5---x7---x3---z1---
                        X
                  ---x2---z0---
        """
        top_graph = b_left_pattern()
        bottom_graph = b_left_pattern(4, 5, 6, 7)
        graph = nx.compose(top_graph, bottom_graph)
        graph.add_edges_from([(7, 3)], type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(graph)),
                             [BLeftMatch(0, 1, 2, 3), BLeftMatch(4, 5, 6, 7)])

    def test_five_way_cross_composition(self):
        """
         ---z0----x2---        ---z8----x10---
                X                     X
         ---z1----x3----z4---x6---z9----x11---
                           X
         ---z12---x14---z5---x7---z16---x18---
                X                     X
         ---z13---x15---       ---z17---x19---
        """
        bottom_left_graph = b_left_pattern()
        center_graph = b_left_pattern(4, 5, 6, 7)
        top_left_graph = b_left_pattern(8, 9, 10, 11)
        bottom_right_graph = b_left_pattern(12, 13, 14, 15)
        top_right_graph = b_left_pattern(16, 17, 18, 19)
        graph = nx.compose_all([bottom_left_graph, center_graph, top_left_graph, bottom_right_graph, top_right_graph])
        graph.add_edges_from([(3, 4), (6, 9), (14, 5), (7, 16)], type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(graph)),
                             [BLeftMatch(0, 1, 2, 3), BLeftMatch(4, 5, 6, 7),
                              BLeftMatch(8, 9, 10, 11),
                              BLeftMatch(12, 13, 14, 15), BLeftMatch(16, 17, 18, 19)])

    def test_five_way_cross_composition_connected(self):
        """
         ---z0----x2--------------z8----x10---
        (        X                     X      )
         ---z1----x3----z4---x6---z9----x11---
                           X
         ---z12---x14---z5---x7---z16---x18---
        (       X                     X       )
         ---z13---x15-------------z17---x19---
        """
        bottom_left_graph = b_left_pattern()
        center_graph = b_left_pattern(4, 5, 6, 7)
        top_left_graph = b_left_pattern(8, 9, 10, 11)
        bottom_right_graph = b_left_pattern(12, 13, 14, 15)
        top_right_graph = b_left_pattern(16, 17, 18, 19)
        graph = nx.compose_all([bottom_left_graph, center_graph, top_left_graph, bottom_right_graph, top_right_graph])
        graph.add_edges_from([(0, 1), (3, 4), (2, 8), (6, 9), (10, 11), (12, 13), (14, 5), (7, 16), (18, 19), (15, 17)],
                             type=S_ETYPE_INDEX)
        self.assertListEqual(list(b_left_matches(graph)),
                             [BLeftMatch(4, 5, 6, 7), BLeftMatch(0, 1, 2, 3),
                              BLeftMatch(12, 13, 14, 15),
                              BLeftMatch(8, 9, 10, 11),
                              BLeftMatch(16, 17, 18, 19)])

    def test_diamond_composition(self):
        """
                  ---z4---x8---
                        X
        ---z0---x2---z5---x9----z12---x14---
              X                     X
        ---z1---x3---z6---x10---z13---x15---
                        X
                  ---z7---x11---
        """
        print('matches = ', list(b_left_matches(diamond_graph())))
        self.assertListEqual(list(b_left_matches(diamond_graph())),
                             [BLeftMatch(0, 1, 2, 3), BLeftMatch(4, 5, 8, 9),
                              BLeftMatch(6, 7, 10, 11),
                              BLeftMatch(12, 13, 14, 15)])

    def test_square(self):
        """
        ---z0---z2---
            |    |
        ---x1---x3---
        """
        self.assertListEqual(list(b_left_matches(square_graph(3, 5, 7, 9))), [])

    def test_square_alternating(self):
        """
        ---x0---z2---
            |    |
        ---z1---x3---
        """
        self.assertListEqual(list(b_left_matches(square_graph_alternating())), [BLeftMatch(1, 2, 0, 3)])


def wrong_degree_no_match_test_graph(rule_mode: RuleMode) -> nx.MultiGraph:
    node_types = rule_mode_to_ntype_indices(rule_mode)
    graph = nx.MultiGraph()
    graph.add_node(1, type=node_types[0], phase=0, degree=2)
    graph.add_node(2, type=node_types[1], phase=0, degree=2)
    graph.add_edge(1, 2, type=S_ETYPE_INDEX)
    return graph


def parallel_edge_no_match_test_graph(rule_mode: RuleMode) -> nx.MultiGraph:
    node_types = rule_mode_to_ntype_indices(rule_mode)
    graph = nx.MultiGraph()
    graph.add_node(23, type=node_types[0], phase=0, degree=3)
    graph.add_node(17, type=node_types[1], phase=0, degree=3)
    graph.add_edge(23, 17, type=S_ETYPE_INDEX)
    graph.add_edge(23, 17, type=H_ETYPE_INDEX)
    return graph


def nonzero_phase_no_match_test_graph(rule_mode: RuleMode) -> nx.MultiGraph:
    node_types = rule_mode_to_ntype_indices(rule_mode)
    graph = nx.MultiGraph()
    graph.add_node(23, type=node_types[0], phase=4, degree=3)
    graph.add_node(17, type=node_types[1], phase=2, degree=3)
    graph.add_edge(23, 17, type=S_ETYPE_INDEX)
    graph.add_edge(23, 17, type=S_ETYPE_INDEX)
    return graph


def hadamard_edge_no_match_test_graph(rule_mode: RuleMode) -> nx.MultiGraph:
    node_types = rule_mode_to_ntype_indices(rule_mode)
    graph = nx.MultiGraph()
    graph.add_node(23, type=node_types[0], phase=0, degree=3)
    graph.add_node(17, type=node_types[1], phase=0, degree=3)
    graph.add_edge(23, 17, type=H_ETYPE_INDEX)
    return graph


def spring_layout_data(graph: nx.MultiGraph) -> Tuple[List[int], Dict[int, Tuple[int, int]]]:
    xs = [x for _, x in graph.nodes(data=COLUMN)]
    max_x = max(xs)
    min_x = min(xs)
    pos = {n: (ndata[COLUMN], ndata[ROW]) for n, ndata in graph.nodes(data=True) if
           ndata[COLUMN] == max_x or ndata[COLUMN] == min_x}
    fixed = list(pos.keys())
    return fixed, pos


def add_layer_data(graph: nx.MultiGraph) -> None:
    for n, ndata in graph.nodes(data=True):
        ndata['layer'] = ndata[COLUMN] / 100


class BRuleRightTest(unittest.TestCase):

    def test_self_match_z(self):
        self.assertListEqual(list(b_right_matches(b_right_pattern())), [BRightMatch(0, 1)])

    def test_line_graph_no_match_z(self):
        self.assertListEqual(list(b_right_matches(wrong_degree_no_match_test_graph(Z_NTYPE_NAME))), [])

    def test_line_graph_no_match_x(self):
        self.assertListEqual(list(b_right_matches(wrong_degree_no_match_test_graph(X_NTYPE_NAME))), [])

    def test_parallel_edge_no_match_z(self):
        self.assertListEqual(list(b_right_matches(parallel_edge_no_match_test_graph(Z_NTYPE_NAME))), [])

    def test_parallel_edge_no_match_x(self):
        self.assertListEqual(list(b_right_matches(parallel_edge_no_match_test_graph(X_NTYPE_NAME))), [])

    def test_nonzero_phase_no_match_z(self):
        self.assertListEqual(list(b_right_matches(nonzero_phase_no_match_test_graph(Z_NTYPE_NAME))), [])

    def test_nonzero_phase_no_match_x(self):
        self.assertListEqual(list(b_right_matches(nonzero_phase_no_match_test_graph(X_NTYPE_NAME))), [])

    def test_hadamard_edge_no_match_z(self):
        self.assertListEqual(list(b_right_matches(hadamard_edge_no_match_test_graph(Z_NTYPE_NAME))), [])

    def test_hadamard_edge_no_match_x(self):
        self.assertListEqual(list(b_right_matches(hadamard_edge_no_match_test_graph(X_NTYPE_NAME))), [])
