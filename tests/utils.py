import torch
from hypothesis import strategies as st
from hypothesis.strategies import composite

from alphazx.diagram.match import NODE_TYPE_INDICES


def zx_diagram_config_st(max_num_qubits: int = 80, max_depth: int = 80):
    @composite
    def inner_zx_match_diagram_st(draw):
        num_qubits = draw(st.integers(2, max_num_qubits))
        depth = draw(st.integers(1, max_depth))
        t_gates = draw(st.booleans())
        return num_qubits, depth, t_gates
    return inner_zx_match_diagram_st()


@composite
def mask_columns_by_value_st(draw):
    shape = draw(st.lists(st.integers(1, 500), min_size=2, max_size=2))
    length = draw(st.integers(0, 1000))
    return torch.randint(0, 500, shape, dtype=torch.int32), torch.randint(0, 1000, [length], dtype=torch.int32)


@composite
def random_node_types_st(draw):
    return torch.Tensor(draw(st.lists(st.sampled_from(NODE_TYPE_INDICES), unique=True, min_size=0, max_size=len(NODE_TYPE_INDICES))))


@composite
def concatenate_by_group_st(draw):
    x = draw(st.lists(st.floats(allow_nan=False), min_size=2))
    current_group = 0
    remaining_elements = len(x)
    index = []
    while remaining_elements > 0:
        group_size = draw(st.integers(min_value=1, max_value=remaining_elements))
        index = index + (group_size * [current_group])
        current_group += 1
        remaining_elements -= group_size
    return torch.tensor(x, dtype=torch.float64), torch.tensor(index)
