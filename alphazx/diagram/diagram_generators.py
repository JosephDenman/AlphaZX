from fractions import Fraction

import networkx as nx
import pyzx
import torch
from torch_geometric.data import HeteroData, Data

from alphazx.diagram.match import BoundaryMatch, FRightZMatch, FRightXMatch
from alphazx.diagram.pyzx_nx_conv import ETYPE, NTYPE, PHASE, nx_remove_position_attributes
from alphazx.diagram.zx_diagram import ZXDiagram
from alphazx.diagram.zx_match_diagram import to_zx_match_diagram, ZXMatchDiagram


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
                               ndata[NTYPE] == FRightZMatch.abbrev and nx_graph.degree(n) == 2 and ndata[PHASE] == 0]
    for n in boundary_zero_z_spiders:
        neighbors = list(nx_graph.neighbors(n))
        nx_graph.remove_node(n)
        nx_graph.add_edge(neighbors[0], neighbors[1])


def node_types_to_strings(nx_graph: nx.MultiGraph) -> None:
    for node in nx_graph.nodes:
        ntype_index = nx_graph.nodes[node][NTYPE]
        if ntype_index == FRightXMatch.index:
            ntype_string = FRightXMatch.abbrev
        elif ntype_index == FRightZMatch.index:
            ntype_string = FRightZMatch.abbrev
        elif ntype_index == BoundaryMatch.index:
            ntype_string = BoundaryMatch.abbrev
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
    nx_remove_position_attributes(nx_graph)
    remove_edge_types(nx_graph)


def phase_denominator(t_gates: bool = True) -> int:
    return 4 if t_gates else 2


def clifford_nx_graph(num_qubits: int, depth: int, t_gates: bool = True) -> nx.MultiGraph:
    # TODO: Add support for H gates
    nx_graph = graph_to_nx_graph(pyzx.generate.cliffords(num_qubits, depth, True, t_gates))
    post_process(nx_graph)
    return nx_graph


def clifford_zx_diagram(num_qubits: int, depth: int, t_gates: bool = True) -> ZXDiagram:
    return ZXDiagram(phase_denominator(t_gates), clifford_nx_graph(num_qubits, depth, t_gates))


def cnot_had_phase_zx_diagram(
    num_qubits: int,
    depth: int,
    p_had: float = 0.2,
    p_t: float = 0.2,
) -> ZXDiagram:
    """Generate a ZX diagram from a random CNOT+Hadamard+Phase circuit.

    Unlike clifford_zx_diagram (which generates a random ZX graph directly),
    this starts from an actual quantum circuit and converts it to a ZX diagram.
    The resulting diagram has realistic structure and typically more T-gates
    available for optimization, making it better suited for training.

    :param num_qubits: Number of qubits.
    :param depth: Number of gates in the circuit.
    :param p_had: Probability of Hadamard gates (default 0.2).
    :param p_t: Probability of T-gates vs other phase gates (default 0.2).
    """
    return ZXDiagram(
        phase_denominator(True),
        nx_c_not_had_phase_graph(num_qubits, depth, p_had, p_t, clifford=False),
    )


def clifford_pyg_hdata_zx_diagram(num_qubits: int, depth: int, t_gates: bool = True,
                                  sort_by_row: bool = False) -> HeteroData:
    return ZXDiagram(phase_denominator(t_gates), clifford_nx_graph(num_qubits, depth, t_gates)).to_pyg_hdata(
        sort_by_row)


def clifford_pyg_zx_diagram(num_qubits: int, depth: int, t_gates: bool = True, sort_by_row: bool = False) -> Data:
    return ZXDiagram(phase_denominator(t_gates), clifford_nx_graph(num_qubits, depth, t_gates)).to_pyg_data(sort_by_row)


def clifford_zx_match_diagram(num_qubits: int, depth: int, t_gates: bool = True) -> ZXMatchDiagram:
    return to_zx_match_diagram(clifford_zx_diagram(num_qubits, depth, t_gates))


def clifford_pyg_zx_match_diagram(num_qubits: int, depth: int, t_gates: bool = True, with_reverse_mapping: bool = False,
                                  sort_by_row: bool = False) -> Data:
    return clifford_zx_match_diagram(num_qubits, depth, t_gates).to_pyg_data(with_reverse_mapping, sort_by_row)


def clifford_pyg_hdata_zx_match_diagram(num_qubits: int, depth: int, t_gates: bool = True,
                                        with_reverse_mapping: bool = False, sort_by_row: bool = False) -> HeteroData:
    return clifford_zx_match_diagram(num_qubits, depth, t_gates).to_pyg_hdata(with_reverse_mapping, sort_by_row)


def stringify(v: object | list | str) -> str:
    if isinstance(v, object):
        return repr(v)
    elif isinstance(v, list):
        return str(v)
    elif isinstance(v, str):
        return v
    elif isinstance(v, torch.Tensor):
        return str(v.tolist())
    else:
        raise Exception(f'Unsupported value {type(v)}')


def gml_clifford_nx_graph(num_qubits: int, depth: int, t_gates: bool, path: str) -> nx.MultiGraph:
    return nx.write_gml(clifford_zx_diagram(num_qubits, depth, t_gates), path, stringify)


def gml_clifford_zx_diagram(num_qubits: int, depth: int, t_gates: bool, path: str) -> ZXDiagram:
    return nx.write_gml(clifford_zx_diagram(num_qubits, depth, t_gates), path, stringify)


def gml_clifford_zx_match_diagram(num_qubits: int, depth: int, t_gates: bool, path: str) -> ZXMatchDiagram:
    return nx.write_gml(clifford_zx_match_diagram(num_qubits, depth, t_gates), path, stringify)


def gml_clifford_pyg_zx_match_diagram(num_qubits: int, depth: int, t_gates: bool, path: str) -> Data:
    return nx.write_gml(clifford_pyg_zx_match_diagram(num_qubits, depth, t_gates), path, stringify)


def gml_clifford_pyg_hdata_zx_match_diagram(num_qubits: int, depth: int, t_gates: bool, path: str) -> HeteroData:
    return nx.write_gml(clifford_pyg_hdata_zx_match_diagram(num_qubits, depth, t_gates), path, stringify)
