from diagram.pyzx_graph_generator import nx_clifford_graph
from diagram.zx_diagram import ZXDiagram

graph_count = 3
num_qubits = 10
depth = 10
max_match_nodes = 4
num_match_types = 10

for _ in range(graph_count):
    nx_graph = nx_clifford_graph(num_qubits, depth)
    matches = list(ZXDiagram(nx_graph).compute_matches())
    sub_g_nodes_list = []
    ys = []
    for match in matches:
        sub_g_nodes_list.append(list(match.nodes) + [-1] * (max_match_nodes - len(match.nodes)))
        zeroes = [0] * num_match_types
        zeroes[match.index] = 1
        ys.append(zeroes)

