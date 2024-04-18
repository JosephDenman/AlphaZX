import torch
import unittest

from torch_geometric.nn import SoftmaxAggregation

from alphazx.models.utils import cat_aggregate

from hypothesis import strategies as st, given
from hypothesis.strategies import composite


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
    def test_cat_aggregation(self, params: tuple[list[float], list[int], int, list[list[float]]]):
        x, index, max_num_elements, expected = params
        x = torch.tensor(x, dtype=torch.float64)
        index = torch.tensor(index)
        expected = torch.tensor(expected, dtype=torch.float64)
        result = cat_aggregate(x, index)
        self.assertTrue(torch.equal(result, expected))

    @given(aggr_params())
    def test_softmax_aggregation(self, params: tuple[list[float], list[int], int, list[list[float]]]):
        x, index, max_num_elements, expected = params
        x = torch.tensor(x, dtype=torch.float64)
        index = torch.tensor(index, dtype=torch.int64)
        result = SoftmaxAggregation().forward(x, index, dim=0)
        print('x = ', x)
        print('index = ', index)
        print('result = ', result)
