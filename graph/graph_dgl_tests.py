import unittest

import dgl
import networkx as nx

from graph_dgl import GraphDGL


class MyTestCase(unittest.TestCase):

    def test_empty_zx_diagram(self):
        try:
            GraphDGL()
        except (Exception,) as e:
            self.fail(f"Unexpected exception {e}")

    def test_add_nodes(self):
        pass

    def test_add_node_throws_with_invalid_data(self):
        pass

    def test_add_edge(self):
        pass

    def test_add_edge_throws_when_node_nonexistent(self):
        pass

    def test_pyzx_generation(self):
        pass

    def test_draw(self):
        pass


if __name__ == '__main__':
    unittest.main()
