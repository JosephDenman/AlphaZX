from fractions import Fraction

import networkx as nx
import pyzx
import torch_geometric as pyg

from diagram.pyzx_nx_conv import nx_to_pyg_hetero, ETYPE, NTYPE, PHASE, Z_NTYPE_NAME, \
    nx_remove_position_attributes, X_NTYPE_INDEX, X_NTYPE_NAME, Z_NTYPE_INDEX, B_NTYPE_INDEX, B_NTYPE_NAME, COLUMN


def graph_to_nx_graph(graph: pyzx.Graph) -> nx.MultiGraph:
    """
    :param graph: A PyZX diagram.
    :return: A NetworkX multi-diagram. The diagram is an undirected diagram. It does not contain backward links
             to denote undirected edges. Convolutional layers must propagate both directions manually.
    """
    return nx.parse_graphml(graph.to_graphml().replace('edge type', ETYPE), node_type=int,
                            edge_key_type=str,
                            force_multigraph=True)


def nx_c_not_had_phase_graph(num_qubits: int, depth: int, p_had: float = 0.2, p_t: float = 0.2,
                             clifford=False) -> nx.MultiGraph:
    nx_graph = graph_to_nx_graph(
        pyzx.generate.CNOT_HAD_PHASE_circuit(num_qubits, depth, p_had, p_t, clifford).to_graph())
    post_process(nx_graph)
    return nx_graph


def node_phases_to_floats(nx_graph: nx.MultiGraph) -> None:
    for node in nx_graph.nodes:
        phase_string = nx_graph.nodes[node][PHASE]
        phase_float = float(Fraction(phase_string))
        nx_graph.nodes[node][PHASE] = phase_float


def remove_boundary_zero_z_spiders(nx_graph: nx.MultiGraph) -> None:
    assert not nx_graph.is_directed(), "Graph must be undirected"
    boundary_zero_z_spiders = [n for n, ndata in nx_graph.nodes(data=True) if
                               ndata[NTYPE] == Z_NTYPE_NAME and nx_graph.degree(n) == 2 and ndata[PHASE] == 0]
    for n in boundary_zero_z_spiders:
        neighbors = list(nx_graph.neighbors(n))
        nx_graph.remove_node(n)
        nx_graph.add_edge(neighbors[0], neighbors[1])


def node_types_to_strings(nx_graph: nx.MultiGraph) -> None:
    for node in nx_graph.nodes:
        ntype_index = nx_graph.nodes[node][NTYPE]
        if ntype_index == X_NTYPE_INDEX:
            ntype_string = X_NTYPE_NAME
        elif ntype_index == Z_NTYPE_INDEX:
            ntype_string = Z_NTYPE_NAME
        elif ntype_index == B_NTYPE_INDEX:
            ntype_string = B_NTYPE_NAME
        else:
            raise Exception(f'Node {node} has unexpected type index {ntype_index}')
        nx_graph.nodes[node][NTYPE] = ntype_string


def remove_edge_types(nx_graph: nx.MultiGraph) -> None:
    for s, t, edata in nx_graph.edges(data=True):
        if ETYPE in edata.keys():
            del edata[ETYPE]


def post_process(nx_graph: nx.MultiGraph) -> None:
    nx_graph.to_undirected()
    node_types_to_strings(nx_graph)
    node_phases_to_floats(nx_graph)
    remove_boundary_zero_z_spiders(nx_graph)
    # nx_remove_position_attributes(nx_graph)
    remove_edge_types(nx_graph)


"""def calculate_input_nodes(nx_graph: nx.MultiGraph) -> None:
    input_node_x_pos = min([ndata[COLUMN] for _, ndata in nx_graph.nodes(data=True)])
    output_node_x_pos = max([ndata[COLUMN] for _, ndata in nx_graph.nodes(data=True)])
    for node, ndata in nx_graph.nodes(data=True):"""


def nx_clifford_graph(num_qubits: int, depth: int,
                      t_gates: bool = True) -> nx.MultiGraph:
    nx_graph = graph_to_nx_graph(pyzx.generate.cliffords(num_qubits, depth, False, t_gates))
    post_process(nx_graph)
    return nx_graph


def pyg_clifford_graph(num_qubits: int, depth: int,
                       t_gates: bool = True) -> pyg.data.HeteroData:
    return nx_to_pyg_hetero(nx_clifford_graph(num_qubits, depth, t_gates), 'type')
