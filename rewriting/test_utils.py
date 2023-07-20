from typing import Optional

from hypothesis import strategies as st
from hypothesis.strategies import SearchStrategy


def st_phase(d: int) -> SearchStrategy[float]:
    return st.sampled_from([n / d for n in range(2*d)])


def st_node(d: Optional[int]) -> SearchStrategy[tuple[int, float]]:
    if d is None:
        return st.tuples(st.integers(0, 2), st.sampled_from([0]))
    else:
        return st.tuples(st.integers(0, 2), st_phase(d))


def st_b_right_nodes() -> SearchStrategy[list[tuple[int, float]]]:
    return st.lists(st_node(None), min_size=4, max_size=4)

