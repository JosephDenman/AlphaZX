from fractions import Fraction

import networkx as nx
import pyzx
import torch_geometric as pyg

from graph.pyzx_nx_conversion import nx_to_pyg_heterograph, ETYPE, NTYPE, PHASE, Z_NTYPE_INDEX, S_ETYPE_INDEX, \
    nx_remove_position_attributes


def graph_to_nx_graph(graph: pyzx.Graph) -> nx.MultiGraph:
    """
    :param graph: A PyZX graph.
    :return: A NetworkX multi-graph. The graph is an undirected graph. It does not contain backward links to denote
             undirected edges. Convolutional layers must propagate both directions manually.
    """
    return nx.parse_graphml(graph.to_graphml().replace('edge type', ETYPE), node_type=int,
                            edge_key_type=int,
                            force_multigraph=True)


def nx_cnot_had_phase_graph(num_qubits: int, depth: int, p_had: float = 0.2, p_t: float = 0.2,
                            clifford=False) -> nx.MultiGraph:
    nx_graph = graph_to_nx_graph(
        pyzx.generate.CNOT_HAD_PHASE_circuit(num_qubits, depth, p_had, p_t, clifford).to_graph())
    post_process(nx_graph)
    return nx_graph


def node_phases_to_ints(nx_graph: nx.MultiGraph) -> None:
    for node in nx_graph.nodes:
        phase_string = nx_graph.nodes[node][PHASE]
        phase_float = float(Fraction(phase_string))
        nx_graph.nodes[node][PHASE] = phase_float


def remove_boundary_zero_z_spiders(nx_graph: nx.MultiGraph) -> None:
    assert not nx_graph.is_directed(), "Graph must be undirected"
    boundary_zero_z_spiders = [n for n, ndata in nx_graph.nodes(data=True) if
                               ndata[NTYPE] == Z_NTYPE_INDEX and nx_graph.degree(n) == 2 and ndata[PHASE] == 0]
    for n in boundary_zero_z_spiders:
        neighbors = list(nx_graph.neighbors(n))
        nx_graph.remove_node(n)
        nx_graph.add_edge(neighbors[0], neighbors[1], type=S_ETYPE_INDEX)


def post_process(nx_graph: nx.MultiGraph) -> None:
    nx_graph.to_undirected()
    node_phases_to_ints(nx_graph)
    remove_boundary_zero_z_spiders(nx_graph)
    nx_remove_position_attributes(nx_graph)


def nx_clifford_graph(num_qubits: int, depth: int, no_hadamard: bool = True,
                      t_gates: bool = True) -> nx.MultiGraph:
    nx_graph = graph_to_nx_graph(pyzx.generate.cliffords(num_qubits, depth, no_hadamard, t_gates))
    post_process(nx_graph)
    return nx_graph


def pyg_clifford_graph(num_qubits: int, depth: int, no_hadamard: bool = True,
                       t_gates: bool = True) -> pyg.data.HeteroData:
    return nx_to_pyg_heterograph(nx_clifford_graph(num_qubits, depth, no_hadamard, t_gates))