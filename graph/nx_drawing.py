from fractions import Fraction
from typing import Tuple, List, Dict, Any

import networkx as nx

from graph.pyzx_nx_dgl_conversion import Z_NTYPE_INDEX, X_NTYPE_INDEX

Z_NODE_COLOR = '#d8f8d8'
X_NODE_COLOR = '#e8a5b0'
H_NODE_COLOR = '#fef9b5'
BOUNDARY_NODE_COLOR = '#000000'
MATCH_Z_NODE = 'white'
MATCH_X_NODE = 'grey'


def node_type_to_color(node_type: int) -> str:
    if node_type is 0:
        return BOUNDARY_NODE_COLOR
    elif node_type is 1:
        return Z_NODE_COLOR
    elif node_type is 2:
        return X_NODE_COLOR
    elif node_type is 3:
        return H_NODE_COLOR
    else:
        raise Exception('Unexpected node type ' + str(node_type))


def node_match_to_color(ndata: Dict[str, Any]) -> str:
    if ndata['type'] == Z_NTYPE_INDEX:
        return MATCH_Z_NODE
    elif ndata['type'] == X_NTYPE_INDEX:
        return MATCH_X_NODE
    else:
        raise Exception('Unexpected node type index ' + str(ndata['type']))


def node_is_zx(node_type: int) -> bool:
    return node_type in [1, 2]


def node_is_boundary(node_type: int) -> bool:
    return node_type is 0


SIMPLE_EDGE_COLOR = '#000000'
H_EDGE_COLOR = '#0000f5'


def edge_type_to_color(edge_type: int) -> str:
    if edge_type is 1:
        return SIMPLE_EDGE_COLOR
    elif edge_type is 2:
        return H_EDGE_COLOR
    else:
        raise Exception('Unexpected edge type ' + str(edge_type))


def edge_is_simple(edge_type: int) -> bool:
    return edge_type is 1


def node_size(node_data: Dict) -> int:
    phase = node_data['phase']
    if node_is_boundary(node_data['type']):
        return 0
    else:
        if phase == 0:
            return 60
        elif phase == 1:
            return 300
        else:
            return len(str(Fraction(phase))) ** 2 * 60


# TODO: Adjust row offsets based on computed node size
def node_styling(graph: nx.Graph, matches: Dict[int, int] = None) -> Tuple[Dict, Dict, List, List]:
    labels = {}
    positions = {}
    sizes = []
    colors = []
    for node, node_data in graph.nodes(data=True):
        if node_is_zx(node_data['type']) and node_data['phase'] != 0:
            labels[node] = node_data['phase']
        positions[node] = [node_data['x'], node_data['y']]
        sizes.append(node_size(node_data))
        if matches is not None and node in matches:
            colors.append(node_match_to_color(node_data))
        else:
            colors.append(node_type_to_color(node_data['type']))
    return labels, positions, sizes, colors


def edge_styling(graph: nx.Graph) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    h_edge_list = []
    simple_edge_list = []
    for *edge, edge_data in graph.edges(data=True):
        [h_edge_list, simple_edge_list][edge_is_simple(edge_data['type'])].append(edge)
    return h_edge_list, simple_edge_list


def draw_nx_zx_diagram(nx_zx_diagram: nx.Graph, matches: Dict[int, int] = None,
                       pos: Dict[int, Tuple[int, int]] = None) -> None:
    node_labels, node_positions, node_sizes, node_colors = node_styling(nx_zx_diagram, matches)
    if pos is not None:
        node_positions = pos
    h_edge_list, simple_edge_list = edge_styling(nx_zx_diagram)
    nx.draw_networkx_nodes(nx_zx_diagram, node_size=node_sizes, node_color=node_colors, pos=node_positions)
    nx.draw_networkx_edges(nx_zx_diagram, edgelist=simple_edge_list, node_size=node_sizes, edge_color=SIMPLE_EDGE_COLOR,
                           pos=node_positions, style='-')
    nx.draw_networkx_edges(nx_zx_diagram, edgelist=h_edge_list, node_size=node_sizes, edge_color=H_EDGE_COLOR,
                           pos=node_positions, style='--')
    nx.draw_networkx_labels(nx_zx_diagram, node_positions, node_labels)
