import unittest

from diagram.zx_diagram import ZXDiagram


class ZXDiagramTest(unittest.TestCase):

    @staticmethod
    def test_simple_add_remove():
        d = ZXDiagram()
        z = d.add_z_node(0.5)
        x = d.add_x_node(-0.5)
        d.add_s_edges_from(2 * [(z, x)])
        d.remove_z_node(z)
