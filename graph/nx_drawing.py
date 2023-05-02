from fractions import Fraction
from typing import Tuple, List, Dict, Any

import networkx as nx

from graph.pyzx_nx_conversion import Z_NTYPE_INDEX, X_NTYPE_INDEX, S_ETYPE_INDEX, H_ETYPE_INDEX, B_NTYPE_INDEX, \
    is_boundary, H_NTYPE_INDEX, is_basis, is_simple_edge, PHASE, NTYPE, ETYPE, ROW, COLUMN
from matching.base import Match

Z_NODE_COLOR = '#d8f8d8'
X_NODE_COLOR = '#e8a5b0'
H_NODE_COLOR = '#fef9b5'
BOUNDARY_NODE_COLOR = '#000000'
MATCH_Z_NODE_COLOR = 'white'
MATCH_X_NODE_COLOR = 'purple'
MATCH_S_EDGE_COLOR = 'red'


def node_type_to_color(ntype: int) -> str:
    if ntype == B_NTYPE_INDEX:
        return BOUNDARY_NODE_COLOR
    elif ntype == Z_NTYPE_INDEX:
        return Z_NODE_COLOR
    elif ntype == X_NTYPE_INDEX:
        return X_NODE_COLOR
    elif ntype == H_NTYPE_INDEX:
        return H_NODE_COLOR
    else:
        raise Exception('Unexpected node type ' + str(ntype))


def node_match_to_color(ndata: Dict[str, Any]) -> str:
    if ndata[NTYPE] == Z_NTYPE_INDEX:
        return MATCH_Z_NODE_COLOR
    elif ndata[NTYPE] == X_NTYPE_INDEX:
        return MATCH_X_NODE_COLOR
    else:
        raise Exception('Unexpected node type index ' + str(ndata[NTYPE]))


SIMPLE_EDGE_COLOR = '#000000'
H_EDGE_COLOR = '#0000f5'
MATCHED_SIMPLE_EDGE_COLOR = 'red'
MATCHED_H_EDGE_COLOR = 'red'


def edge_type_to_color(etype: int) -> str:
    if etype == S_ETYPE_INDEX:
        return SIMPLE_EDGE_COLOR
    elif etype == H_ETYPE_INDEX:
        return H_EDGE_COLOR
    else:
        raise Exception('Unexpected edge type ' + str(etype))


def node_size(ndata: Dict) -> int:
    phase = ndata[PHASE]
    if is_boundary(ndata[NTYPE]):
        return 0
    else:
        if phase == 0:
            return 60
        elif phase == 1:
            return 300
        else:
            return len(str(Fraction(phase))) ** 2 * 60


# TODO: Adjust row offsets based on computed node size
def node_styling(nx_graph: nx.MultiGraph) -> Tuple[Dict, Dict, List, List]:
    labels = {}
    positions = {}
    sizes = []
    colors = []
    for node, node_data in nx_graph.nodes(data=True):
        if is_basis(node_data[NTYPE]) and node_data[PHASE] != 0:
            labels[node] = node_data[PHASE]
        positions[node] = [node_data[ROW], node_data[COLUMN]]
        sizes.append(node_size(node_data))
        colors.append(node_type_to_color(node_data[NTYPE]))
    return labels, positions, sizes, colors

EdgeList = List[Tuple[int, int]]

def edge_styling(nx_graph: nx.MultiGraph, match: nx.MultiGraph) -> Tuple[EdgeList, EdgeList, EdgeList, EdgeList]:
    h_edge_list = []
    simple_edge_list = []
    matched_h_edge_list = []
    matched_simple_edge_list = []
    for *edge, edge_data in nx_graph.edges(data=True):
        ([matched_h_edge_list, matched_simple_edge_list] if match.has_edge(*edge) else [h_edge_list,
                                                                                        simple_edge_list])[
            is_simple_edge(edge_data[ETYPE])].append(edge)
    return h_edge_list, simple_edge_list, matched_h_edge_list, matched_simple_edge_list


def draw_nx_zx_diagram(nx_graph: nx.MultiGraph, match: Match = None,
                       pos: Dict[int, Tuple[int, int]] = None) -> None:
    node_labels, node_positions, node_sizes, node_colors = node_styling(nx_graph)
    matched_subgraph = nx_graph.subgraph(match)
    if pos is not None:
        node_positions = pos
    h_edge_list, simple_edge_list, matched_h_edge_list, matched_simple_edge_list = edge_styling(nx_graph,
                                                                                                matched_subgraph)
    nx.draw_networkx_nodes(nx_graph, node_size=node_sizes, node_color=node_colors, pos=node_positions)
    nx.draw_networkx_edges(nx_graph, edgelist=h_edge_list, node_size=node_sizes, edge_color=H_EDGE_COLOR,
                           pos=node_positions, style='--')
    nx.draw_networkx_edges(nx_graph, edgelist=simple_edge_list, node_size=node_sizes, edge_color=SIMPLE_EDGE_COLOR,
                           pos=node_positions, style='-')
    nx.draw_networkx_edges(nx_graph, edgelist=matched_h_edge_list, node_size=node_sizes,
                           edge_color=MATCHED_H_EDGE_COLOR,
                           pos=node_positions, style='--')
    nx.draw_networkx_edges(nx_graph, edgelist=matched_simple_edge_list, node_size=node_sizes,
                           edge_color=MATCHED_SIMPLE_EDGE_COLOR,
                           pos=node_positions, style='-')
    nx.draw_networkx_labels(nx_graph, node_positions, node_labels)
