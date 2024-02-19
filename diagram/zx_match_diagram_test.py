import unittest

from diagram.pyzx_graph_generator import clifford_pyg_hetero_zx_match_diagram, clifford_pyg_zx_match_diagram

num_qubits = 10
depth = 10
t_gates = True
one_hot_types = True
phase_denominator = 8

d = clifford_pyg_zx_match_diagram(num_qubits, depth, t_gates, one_hot_types)
print(d)