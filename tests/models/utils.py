import unittest

import torch
from hypothesis import given, settings

from alphazx.models.utils import concatenate_by_group, concatenate_neighbor_features, concatenate_with_neighbor_features, \
    mask_non_basis_edges, edge_index_as_node_types, compute_column_mask_for_values, mask_edges_by_type
from alphazx.diagram.diagram_generators import clifford_zx_match_diagram
from alphazx.diagram.match import METADATA
from tests.utils import zx_diagram_config_st, mask_columns_by_value_st, random_node_types_st, concatenate_by_group_st


def concatenate_by_group_py(x, index) -> torch.Tensor:
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
        group.extend([-torch.inf] * (max_num_elements - len(group)))
    return torch.tensor(groups, dtype=torch.float64)


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
            expected_list[i].append([0., 0.])
    return torch.tensor(expected_list)


def concatenate_with_neighbor_features_py(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    x_ = concatenate_neighbor_features_py(x, edge_index)
    expected_list = []
    for i, x_n in enumerate(x):
        expected_list.append([x_n.tolist()] + x_[i].tolist())
    return torch.tensor(expected_list)


def edge_index_as_node_types_py(edge_index: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
    edge_index = edge_index.T.tolist()
    filtered_edge_index = []
    for i, edge in enumerate(edge_index):
        filtered_edge_index.append([node_types[edge[0]], node_types[edge[1]]])
    return torch.tensor(filtered_edge_index).T


def compute_column_mask_for_values_py(t: torch.Tensor, values_to_mask: torch.Tensor) -> torch.Tensor:
    if len(values_to_mask) == 0:
        return t
    values_to_mask = set(values_to_mask.tolist())
    result = []
    for i in range(t.shape[1]):
        column = t[:, i].tolist()
        column_set = set(column)
        result.append(len(column_set.intersection(values_to_mask)) == 0)
    result = torch.tensor(result, dtype=torch.bool)
    return result


def mask_edges_by_type_py(edge_index: torch.Tensor, node_types: torch.Tensor,
                          node_types_to_mask: torch.Tensor) -> torch.Tensor:
    column_mask = compute_column_mask_for_values_py(edge_index_as_node_types_py(edge_index, node_types),
                                                    node_types_to_mask)
    return edge_index[:, column_mask]


def mask_non_basis_edges_py(edge_index: torch.Tensor, node_types: torch.Tensor) -> torch.Tensor:
    return mask_edges_by_type_py(edge_index, node_types, torch.tensor(METADATA.non_basis_node_type_indices))


class UtilsTest(unittest.TestCase):
    @given(concatenate_by_group_st())
    def test_concat_group_by(self, params: tuple[torch.Tensor, torch.Tensor]):
        x, index = params
        actual = concatenate_by_group(x, index)
        expected = concatenate_by_group_py(x, index)
        self.assertTrue(torch.equal(actual, expected))

    @given(zx_diagram_config_st(30, 30))
    def test_concatenate_neighbor_features(self, config: tuple[int, int, bool]):
        d = clifford_zx_match_diagram(*config[:-1]).to_pyg_data()
        x = d.x
        edge_index = d.edge_index
        actual = concatenate_neighbor_features(x, edge_index)
        expected = concatenate_neighbor_features_py(x, edge_index)
        self.assertTrue(torch.equal(actual, expected))

    @given(zx_diagram_config_st(30, 30))
    def test_concatenate_with_neighbor_features(self, config: tuple[int, int, bool]):
        d = clifford_zx_match_diagram(*config[:-1]).to_pyg_data()
        x = d.x
        edge_index = d.edge_index
        actual = concatenate_with_neighbor_features(x, edge_index)
        expected = concatenate_with_neighbor_features_py(x, edge_index)
        self.assertTrue(torch.equal(actual, expected))

    @given(zx_diagram_config_st())
    def test_mask_non_basis_edges(self, config: tuple[int, int, bool]):
        d = clifford_zx_match_diagram(*config[:-1]).to_pyg_data()
        edge_index = d.edge_index
        node_types = d.node_type
        actual = mask_non_basis_edges(edge_index, node_types)
        expected = mask_non_basis_edges_py(edge_index, node_types)
        self.assertTrue(torch.equal(actual, expected))

    @given(zx_diagram_config_st(30, 30), random_node_types_st())
    def test_mask_edges_by_type(self, config: tuple[int, int, bool], node_types_to_mask: torch.Tensor):
        d = clifford_zx_match_diagram(*config[:-1]).to_pyg_data()
        edge_index = d.edge_index
        node_types = d.node_type
        actual = mask_edges_by_type(edge_index, node_types, node_types_to_mask)
        expected = mask_edges_by_type_py(edge_index, node_types, node_types_to_mask)
        self.assertTrue(torch.equal(actual, expected))

    @given(zx_diagram_config_st())
    def test_edge_index_as_node_types(self, config: tuple[int, int, bool]):
        d = clifford_zx_match_diagram(*config[:-1]).to_pyg_data()
        edge_index = d.edge_index
        node_types = d.node_type
        actual = edge_index_as_node_types(edge_index, node_types)
        expected = edge_index_as_node_types_py(edge_index, node_types)
        self.assertTrue(torch.equal(actual, expected))

    @given(mask_columns_by_value_st())
    def test_mask_columns_with_values(self, config: tuple[torch.Tensor, torch.Tensor]):
        t, values_to_mask = config
        actual = compute_column_mask_for_values(t, values_to_mask)
        expected = compute_column_mask_for_values_py(t, values_to_mask)
        self.assertTrue(torch.equal(actual, expected))
