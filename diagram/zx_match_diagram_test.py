import networkx as nx
import matplotlib.pyplot as plt
from diagram.diagram_generators import clifford_zx_match_diagram, clifford_zx_diagram
from diagram.nx_drawing import draw_nx_zx_diagram


num_qubits = 10
depth = 10
t_gates = True
one_hot_types = True

d = clifford_zx_diagram(num_qubits, depth, t_gates)
z0 = d.add_z_node(0.)
d.nodes[z0]['x'] = -200.
d.nodes[z0]['y'] = 0.
z1 = d.add_z_node(0.)
d.nodes[z1]['x'] = -400.
d.nodes[z1]['y'] = 0.
d.add_s_edge(z0, z1)

b_nodes = d.b_nodes()
d_copy = d.copy()

for c in nx.connected_components(d_copy):
    if b_nodes.isdisjoint(c):
        d.remove_nodes_from(c)

print(d.node_attrs)
plt.figure('There')
draw_nx_zx_diagram(d_copy)

plt.figure('Hello')
draw_nx_zx_diagram(d)
plt.show()
