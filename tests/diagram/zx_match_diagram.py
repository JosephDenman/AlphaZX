import unittest

import torch_geometric.data as pyg_data
from hypothesis import given

from alphazx.diagram.diagram_generators import clifford_zx_match_diagram
from tests.utils import zx_diagram_config_st


class TestZXMatchDiagram(unittest.TestCase):
    @given(zx_diagram_config_st())
    def test_zx_match_diagram_to_hdata(self, config: tuple[int, int, bool]):
        result = clifford_zx_match_diagram(*config[:-1]).to_pyg_hdata()
        assert isinstance(result, pyg_data.HeteroData)

    @given(zx_diagram_config_st())
    def test_zx_match_diagram_to_data(self, config: tuple[int, int, bool]):
        result = clifford_zx_match_diagram(*config[:-1]).to_pyg_data()
        assert isinstance(result, pyg_data.Data)
