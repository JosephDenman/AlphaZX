from fractions import Fraction
from typing import Tuple, List, Dict, Optional

import networkx as nx

from diagram.pyzx_nx_conv import Z_NTYPE_INDEX, X_NTYPE_INDEX, S_ETYPE_INDEX, H_ETYPE_INDEX, B_NTYPE_INDEX, \
    is_boundary, H_NTYPE_INDEX, is_basis, is_simple_edge, PHASE, NTYPE, ETYPE, ROW, COLUMN
from matching.match import Match

Z_NTYPE_COLOR = '#d8f8d8'
X_NTYPE_COLOR = '#e8a5b0'
H_NTYPE_COLOR = '#fef9b5'
B_NTYPE_COLOR = '#000000'
NODE_BORDER_COLOR = '#000000'
MATCHED_Z_NTYPE_COLOR = '#ffffff'
MATCHED_X_NTYPE_COLOR = '#b4b4b4'


def node_type_to_color(ntype: int) -> str:
    if ntype == B_NTYPE_INDEX:
        return B_NTYPE_COLOR
    elif ntype == Z_NTYPE_INDEX:
        return Z_NTYPE_COLOR
    elif ntype == X_NTYPE_INDEX:
        return X_NTYPE_COLOR
    elif ntype == H_NTYPE_INDEX:
        return H_NTYPE_COLOR
    else:
        raise Exception('Unexpected node type ' + str(ntype))


def node_match_to_color(ntype: int) -> str:
    if ntype == Z_NTYPE_INDEX:
        return MATCHED_Z_NTYPE_COLOR
    elif ntype == X_NTYPE_INDEX:
        return MATCHED_X_NTYPE_COLOR
    else:
        raise Exception('Unexpected node type index ' + str(ntype))


S_ETYPE_COLOR = '#000000'
H_ETYPE_COLOR = '#0000f5'
MATCHED_S_ETYPE_COLOR = '#ffcc00'
MATCHED_H_ETYPE_COLOR = '#ffcc00'


def edge_type_to_color(etype: int) -> str:
    if etype == S_ETYPE_INDEX:
        return S_ETYPE_COLOR
    elif etype == H_ETYPE_INDEX:
        return H_ETYPE_COLOR
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


def subgraph_from_match(nx_graph: nx.MultiGraph, match: Match) -> nx.MultiGraph:
    return nx_graph.subgraph(match)


# TODO: Adjust row offsets based on computed node size
def node_styling(nx_graph: nx.MultiGraph, match: Optional[Match] = None) -> Tuple[Dict, Dict, List, List, List]:
    labels = {}
    positions = {}
    sizes = []
    colors = []
    border_colors = [NODE_BORDER_COLOR] * nx_graph.number_of_nodes()
    matched_subgraph = subgraph_from_match(nx_graph, match) if match is not None else None
    for n, ndata in nx_graph.nodes(data=True):
        if is_basis(ndata[NTYPE]) and ndata[PHASE] != 0:
            labels[n] = ndata[PHASE]
        positions[n] = [ndata[ROW], ndata[COLUMN]]
        sizes.append(node_size(ndata))
        colors.append(node_match_to_color(ndata[NTYPE]) if matched_subgraph is not None and matched_subgraph.has_node(
            n) else node_type_to_color(ndata[NTYPE]))
    return labels, positions, sizes, colors, border_colors


EdgeList = List[Tuple[int, int]]


# TODO: Fix to work with updated Match structure

def edge_styling(nx_graph: nx.MultiGraph, match: Optional[Match] = None) -> Tuple[
        EdgeList, EdgeList, EdgeList, EdgeList]:
    matched_subgraph = subgraph_from_match(nx_graph, match) if match is not None else None
    h_edge_list = []
    simple_edge_list = []
    matched_h_edge_list = []
    matched_simple_edge_list = []
    for *edge, edge_data in nx_graph.edges(data=True):
        ([matched_h_edge_list, matched_simple_edge_list] if matched_subgraph is not None and matched_subgraph.has_edge(
            *edge) else [h_edge_list,
                         simple_edge_list])[
            is_simple_edge(edge_data[ETYPE])].append(edge)
    return h_edge_list, simple_edge_list, matched_h_edge_list, matched_simple_edge_list


# TODO - Draw groups of edges between two nodes as a single edge with a number label indicating the true number of edges
#        between the vertices.
def draw_nx_zx_diagram(nx_graph: nx.MultiGraph, match: Optional[Match] = None,
                       pos: Optional[Dict[int, Tuple[int, int]]] = None) -> None:
    node_labels, node_positions, node_sizes, node_colors, node_border_colors = node_styling(nx_graph, match)
    node_positions = pos if pos is not None else node_positions
    h_edge_list, simple_edge_list, matched_h_edge_list, matched_simple_edge_list = edge_styling(nx_graph, match)
    nx.draw_networkx_nodes(nx_graph, node_size=node_sizes, node_color=node_colors, pos=node_positions,
                           edgecolors=node_border_colors)
    nx.draw_networkx_edges(nx_graph, edgelist=h_edge_list, node_size=node_sizes, edge_color=H_ETYPE_COLOR,
                           pos=node_positions, style='--')
    nx.draw_networkx_edges(nx_graph, edgelist=simple_edge_list, node_size=node_sizes, edge_color=S_ETYPE_COLOR,
                           pos=node_positions, style='-')
    nx.draw_networkx_edges(nx_graph, edgelist=matched_h_edge_list, node_size=node_sizes,
                           edge_color=MATCHED_H_ETYPE_COLOR,
                           width=2.0,
                           pos=node_positions, style='--')
    nx.draw_networkx_edges(nx_graph, edgelist=matched_simple_edge_list, node_size=node_sizes,
                           edge_color=MATCHED_S_ETYPE_COLOR,
                           width=2.0,
                           pos=node_positions, style='-')
    nx.draw_networkx_labels(nx_graph, node_positions, node_labels)
