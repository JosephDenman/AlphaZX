from diagram.pyzx_graph_generator import clifford_zx_match_diagram

num_qubits = 10
depth = 10
t_gates = True
one_hot_types = True

d = clifford_zx_match_diagram(num_qubits, depth, t_gates, one_hot_types)
pyg_d = d.to_pyg_data()
