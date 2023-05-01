import unittest

import dgl
import networkx as nx

from zx_diagram import NodeType, ZXDiagram, node_is_zx, EdgeType


class MyTestCase(unittest.TestCase):
    def test_is_zx(self):
        self.assertTrue(node_is_zx(NodeType.Z))
        self.assertTrue(node_is_zx(NodeType.X))
        self.assertFalse(node_is_zx(NodeType.BOUNDARY))

    def test_empty_zx_diagram(self):
        zx_diagram = ZXDiagram()
        self.assertEqual(zx_diagram.num_nodes(), 0)
        self.assertEqual(zx_diagram.num_edges(), 0)

    def test_add_nodes(self):
        zx_diagram = ZXDiagram()
        zx_diagram.add_z_node(1.)
        zx_diagram.add_x_node(0.)
        print(zx_diagram.x_nodes())
        zx_diagram.add_boundary_node()
        self.assertEqual(zx_diagram.num_z_nodes(), 1)
        self.assertEqual(zx_diagram.num_x_nodes(), 1)
        self.assertEqual(zx_diagram.num_boundary_nodes(), 1)

    def test_add_node_throws_with_invalid_data(self):
        pass

    def test_add_edge(self):
        zx_diagram = ZXDiagram()
        zx_diagram.add_x_node(0.)
        zx_diagram.add_x_node(0.5)
        zx_diagram.add_z_node(1.)
        zx_diagram.add_z_node(1.5)
        zx_diagram.add_boundary_node()
        zx_diagram.add_boundary_node()
        zx_diagram.add_z_x_hadamard_edge(0, 1)
        zx_diagram.add_z_x_hadamard_edge(0, 1)
        zx_diagram.add_boundary_node()
        zx_diagram.add_boundary_node()
        zx_diagram.add_boundary_boundary_hadamard_edge(0, 1)
        print(zx_diagram.z_x_hadamard_edges())
        print(zx_diagram.boundary_boundary_hadamard_edges())

    def test_add_edge_throws_when_node_nonexistent(self):
        pass

    def test_pyzx_generation(self):
        import pyzx as zx
        import matplotlib.pyplot as plt
        num_qubits = 15
        depth = 15
        circuit = zx.generate.cliffordT(num_qubits, depth)
        zx.draw(circuit)
        plt.show()

    def test_draw(self):
        zx_diagram = ZXDiagram()
        zx_diagram.add_boundary_node()
        zx_diagram.add_z_node(0.)
        zx_diagram.add_x_node(1.)
        zx_diagram.add_z_boundary_hadamard_edge(0, 0)
        zx_diagram.add_x_boundary_simple_edge(0, 0)
        zx_diagram.add_z_x_hadamard_edge(0, 0)
        undirected_nx_graph = zx_diagram.to_undirected_nx_graph()
        import networkx as nx
        import matplotlib.pyplot as plt
        nx.draw(undirected_nx_graph)
        plt.show()


if __name__ == '__main__':
    unittest.main()
