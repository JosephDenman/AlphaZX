import unittest

import torch_geometric.data as pyg_data
from hypothesis import strategies as st, given
from hypothesis.strategies import composite

from alphazx.diagram.diagram_generators import clifford_zx_match_diagram


@composite
def zx_match_diagram_st(draw):
    num_qubits = draw(st.integers(2, 100))
    depth = draw(st.integers(1, 100))
    t_gates = draw(st.booleans())
    with_reverse_mapping = draw(st.booleans())
    return num_qubits, depth, t_gates, with_reverse_mapping


class TestZXMatchDiagram(unittest.TestCase):
    @given(zx_match_diagram_st())
    def test_zx_match_diagram_to_hdata(self, config: tuple[int, int, bool, bool]):
        result = clifford_zx_match_diagram(*config[:-1]).to_pyg_hdata(with_reverse_mapping=config[-1])
        if isinstance(result, tuple):
            assert isinstance(result[0], pyg_data.HeteroData)

    @given(zx_match_diagram_st())
    def test_zx_match_diagram_to_data(self, config: tuple[int, int, bool, bool]):
        result = clifford_zx_match_diagram(*config[:-1]).to_pyg_data(with_reverse_mapping=config[-1])
        if isinstance(result, tuple):
            assert isinstance(result[0], pyg_data.Data)
