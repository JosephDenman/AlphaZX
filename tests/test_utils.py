import unittest

import torch
import numpy as np
from hypothesis import given

from alphazx.diagram.diagram_generators import clifford_zx_match_diagram
from alphazx.diagram.match import METADATA
from alphazx.models.utils import concatenate_neighbor_features, \
    concatenate_with_neighbor_features, \
    mask_non_basis_edges, edge_index_as_node_types, compute_column_mask_for_values, mask_edges_by_type, throw_on_nan
from tests.utils import zx_diagram_config_st, mask_columns_by_value_st, random_node_types_st


def concatenate_by_group_py(x: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
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


def concatenate_neighbor_features_py(x: np.ndarray, edge_index: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    # Initialize lists to hold the batches
    dense_x = []
    dense_mask = []
    # Get the unique nodes (targets) in edge_index[1] (i.e., the destination nodes)
    unique_nodes = np.unique(edge_index[1])
    # For each unique node, collect its neighbor features
    for node in unique_nodes:
        # Find indices in edge_index[1] where the target node is the current node
        neighbor_indices = np.where(edge_index[1] == node)[0]
        # Get the corresponding source nodes from edge_index[0]
        source_nodes = edge_index[0][neighbor_indices]
        # Collect the features of the source nodes
        neighbor_features = x[source_nodes]
        # Store in the dense_x list
        dense_x.append(neighbor_features)
        # Create a mask that indicates which entries in this row are valid
        dense_mask.append(np.ones(len(neighbor_features)))
    # Determine the maximum number of neighbors any node has
    max_len = max(len(neigh) for neigh in dense_x)
    # Pad all lists to the maximum length
    padded_dense_x = np.array(
        [np.pad(neigh, ((0, max_len - len(neigh)), (0, 0)), mode='constant') for neigh in dense_x])
    padded_dense_mask = np.array([np.pad(mask, (0, max_len - len(mask)), mode='constant') for mask in dense_mask])
    return torch.tensor(padded_dense_x, dtype=torch.float64), torch.tensor(padded_dense_mask, dtype=torch.bool)


def concatenate_with_neighbor_features_py(x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
    x_ = concatenate_neighbor_features_py(x.numpy(), edge_index.numpy())[0]
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

    @given(zx_diagram_config_st(30, 30))
    def test_concatenate_neighbor_features(self, config: tuple[int, int, bool]):
        d = clifford_zx_match_diagram(*config[:-1]).to_pyg_data()
        x = d.x
        edge_index = d.edge_index
        actual = concatenate_neighbor_features(x, edge_index)[0]
        expected = concatenate_neighbor_features_py(x.numpy(), edge_index.numpy())[0]
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

    def test_throw_on_nan(self):
        try:
            t = torch.tensor([[0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.6002, 0.0000,
                               0.0000, 0.0000, 0.3998, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                              [torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan,
                               torch.nan,
                               torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan,
                               torch.nan,
                               torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan],
                              [0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.5004, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.4996, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000],
                              [torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan,
                               torch.nan,
                               torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan,
                               torch.nan,
                               torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan, torch.nan],
                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.3361, 0.0000, 0.0000, 0.0000, 0.0000, 0.6639],
                              [0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 1.0000, 0.0000, 0.0000,
                               0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000]], dtype=torch.float64)
            throw_on_nan(t)
            self.fail('Expected error')
        except AssertionError:
            pass
