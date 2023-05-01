import timeit

import dgl
import pyzx
import torch_geometric as pyg
import networkx as nx
from matplotlib import pyplot as plt

from graph.nx_drawing import draw_nx_zx_diagram
from graph.pyzx_nx_dgl_conversion import nx_to_dgl_heterograph, nx_to_pyg_heterograph, add_degree, node_phases_to_ints, \
    add_connected_to, remove_attributes


def graph_to_nx_graph(graph: pyzx.Graph) -> nx.MultiGraph:
    """
    :param graph: A PyZX graph.
    :return: A NetworkX multi-graph. The graph is an undirected graph. It does not contain backward links to denote
             undirected edges. Convolutional layers must propagate both directions manually.
    """
    return nx.parse_graphml(graph.to_graphml().replace('edge type', 'type'), node_type=int,
                            edge_key_type=int,
                            force_multigraph=True)


def nx_cnot_had_phase_graph(num_qubits: int, depth: int, p_had: float = 0.2, p_t: float = 0.2,
                            clifford=False) -> nx.MultiGraph:
    nx_graph = graph_to_nx_graph(
        pyzx.generate.CNOT_HAD_PHASE_circuit(num_qubits, depth, p_had, p_t, clifford).to_graph())
    post_process(nx_graph)
    return nx_graph


def remove_boundary_zero_z_spiders(nx_graph: nx.MultiGraph) -> None:
    assert not nx_graph.is_directed(), "Graph must be undirected"
    boundary_zero_z_spiders = [n for n, ndata in nx_graph.nodes(data=True) if
                               ndata['type'] == 1 and ndata['degree'] == 2 and ndata['phase'] == 0]
    for n in boundary_zero_z_spiders:
        neighbors = list(nx_graph.neighbors(n))
        nx_graph.remove_node(n)
        nx_graph.add_edge(neighbors[0], neighbors[1], type=1)


def post_process(nx_graph: nx.MultiGraph) -> None:
    nx_graph.to_undirected()
    add_degree(nx_graph)
    add_connected_to(nx_graph)
    node_phases_to_ints(nx_graph)
    #remove_boundary_zero_z_spiders(nx_graph)
    #remove_attributes(nx_graph)


def nx_clifford_graph(num_qubits: int, depth: int, no_hadamard: bool = False,
                      t_gates: bool = False) -> nx.MultiGraph:
    nx_graph = graph_to_nx_graph(pyzx.generate.cliffordT(num_qubits, depth, no_hadamard, t_gates))
    post_process(nx_graph)
    return nx_graph


def dgl_cnot_had_phase_graph(num_qubits: int, depth: int, p_had: float = 0.2, p_t: float = 0.2,
                             clifford=False) -> dgl.DGLGraph:
    return nx_to_dgl_heterograph(nx_cnot_had_phase_graph(num_qubits, depth, p_had, p_t, clifford))


def dgl_clifford_graph(num_qubits: int, depth: int, no_hadamard: bool = False, t_gates: bool = False) -> dgl.DGLGraph:
    return nx_to_dgl_heterograph(nx_clifford_graph(num_qubits, depth, no_hadamard, t_gates))


def pyg_clifford_graph(num_qubits: int, depth: int, no_hadamard: bool = True,
                       t_gates: bool = True) -> pyg.data.HeteroData:
    return nx_to_pyg_heterograph(nx_clifford_graph(num_qubits, depth, no_hadamard, t_gates))


#nx_graph = nx_clifford_graph(10, 10)

#pos = nx.nx_pydot.pydot_layout(nx_graph, prog='dot')
#draw_nx_zx_diagram(nx_graph, None, pos)
#plt.show()
