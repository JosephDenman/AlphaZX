import unittest

import torch
from hypothesis import strategies as st, given
from hypothesis.strategies import composite

from alphazx import concatenate_by_group, concatenate_neighbor_features, concatenate_with_neighbor_features
from alphazx.diagram.diagram_generators import clifford_zx_match_diagram
from tests.utils import zx_match_diagram_st


def aggregate_py(x, index) -> tuple[int, list[list[float]]]:
    # Determine the number of groups
    num_groups = max(index) + 1
    # Initialize groups as lists inside a list
    groups = [[] for _ in range(num_groups)]
    # Fill groups based on index
    for val, idx in zip(x, index):
        groups[idx].append(val)
    # Automatically determine the maximum size of any group
    max_num_elements = max(len(group) for group in groups)
    # Pad each group with zeros to make them all the same size
    for group in groups:
        group.extend([0] * (max_num_elements - len(group)))
    return max_num_elements, groups


def concatenate_neighbor_features_py(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    max_nodes = torch.bincount(edge_index[1]).max()
    num_groups = torch.max(edge_index[1]) + 1
    expected_list = []
    for i in range(num_groups):
        expected_list.append([])
    for i, node in enumerate(edge_index[1]):
        expected_list[node.item()].append(x[edge_index[0][i]].tolist())
    for i in range(num_groups):
        while len(expected_list[i]) < max_nodes:
            expected_list[i].append([0, 0])
    return torch.tensor(expected_list)


def concatenate_with_neighbor_features_py(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    x_ = concatenate_neighbor_features_py(x, edge_index)
    expected_list = []
    for i, x_n in enumerate(x):
        expected_list.append([x_n.tolist()] + x_[i].tolist())
    return torch.tensor(expected_list)


@composite
def aggr_params(draw):
    x = draw(st.lists(st.floats(allow_nan=False), min_size=2))
    current_group = 0
    remaining_elements = len(x)
    index = []
    while remaining_elements > 0:
        group_size = draw(st.integers(min_value=1, max_value=remaining_elements))
        index = index + (group_size * [current_group])
        current_group += 1
        remaining_elements -= group_size
    max_num_elements, expected = aggregate_py(x, index)
    return x, index, max_num_elements, expected


class UtilsTest(unittest.TestCase):
    @given(aggr_params())
    def test_concat_group_by(self, params: tuple[list[float], list[int], int, list[list[float]]]):
        x, index, max_num_elements, expected = params
        x = torch.tensor(x, dtype=torch.float64)
        index = torch.tensor(index)
        expected = torch.tensor(expected, dtype=torch.float64)
        result = concatenate_by_group(x, index)
        self.assertTrue(torch.equal(result, expected))

    @given(zx_match_diagram_st())
    def test_concatenate_neighbor_features(self, config: tuple[int, int, bool]):
        d = clifford_zx_match_diagram(*config[:-1]).to_pyg_data()
        x = d.x
        edge_index = d.edge_index
        expected = concatenate_neighbor_features_py(x, edge_index)
        actual = concatenate_neighbor_features(x, edge_index)
        self.assertTrue(torch.all(actual.eq(expected)))

    @given(zx_match_diagram_st())
    def test_concatenate_with_neighbor_features(self, config: tuple[int, int, bool]):
        d = clifford_zx_match_diagram(*config[:-1]).to_pyg_data()
        x = d.x
        edge_index = d.edge_index
        expected = concatenate_with_neighbor_features_py(x, edge_index)
        actual = concatenate_with_neighbor_features(x, edge_index)
        self.assertTrue(torch.all(actual.eq(expected)))
