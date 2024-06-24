import unittest

from hypothesis import given

from alphazx.diagram import clifford_zx_diagram
from tests.utils import zx_diagram_config_st


class TestZXDiagram(unittest.TestCase):
    @given(zx_diagram_config_st())
    def test_zx_diagram_copy(self, config: tuple[int, int, bool]):
        d = clifford_zx_diagram(*config[:-1])
        d_copy = d.copy()
        self.assertSetEqual(d.z_nodes(), d_copy.z_nodes())
        self.assertSetEqual(d.x_nodes(), d_copy.x_nodes())
        self.assertSetEqual(d.b_nodes(), d_copy.b_nodes())
        self.assertEqual(d.phase_denominator, d_copy.phase_denominator)
        self.assertEqual(d.next_node_index, d_copy.next_node_index)
        self.assertEqual(len(d.nodes), len(d_copy.nodes))
        for (n, ndata), (n_copy, ndata_copy) in zip(d.nodes(data=True), d_copy.nodes(data=True)):
            self.assertEqual(n, n_copy)
            self.assertDictEqual(ndata, ndata_copy)
        self.assertEqual(len(d.edges), len(d_copy.edges))
        for (s, t, k, edata), (s_copy, t_copy, k_copy, edata_copy) in zip(d.edges(data=True, keys=True), d_copy.edges(data=True, keys=True)):
            self.assertEqual(s, s_copy)
            self.assertEqual(t, t_copy)
            self.assertEqual(k, k_copy)
            self.assertDictEqual(edata, edata_copy)
        self.assertEqual(d.id, d_copy.id)
