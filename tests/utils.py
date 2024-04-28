from hypothesis import strategies as st
from hypothesis.strategies import composite


@composite
def zx_match_diagram_st(draw):
    num_qubits = draw(st.integers(2, 100))
    depth = draw(st.integers(1, 100))
    t_gates = draw(st.booleans())
    with_reverse_mapping = draw(st.booleans())
    return num_qubits, depth, t_gates, with_reverse_mapping