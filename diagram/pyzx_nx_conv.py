from collections import defaultdict
from typing import Optional, Iterable, Union, List, Any

import networkx as nx
import torch
import torch_geometric as pyg
from torch_geometric.data import HeteroData

B_NTYPE_INDEX = 0
B_NTYPE_NAME = 'b_node'
Z_NTYPE_INDEX = 1
Z_NTYPE_NAME = 'z_node'
X_NTYPE_INDEX = 2
X_NTYPE_NAME = 'x_node'
H_NTYPE_INDEX = 3
H_NTYPE_NAME = 'hadamard'
D_ETYPE_INDEX = 0
D_ETYPE_NAME = 'dummy'
S_ETYPE_INDEX = 1
S_ETYPE_NAME = 'simple'
H_ETYPE_INDEX = 2
H_ETYPE_NAME = 'hadamard'

NTYPE_NAMES = [B_NTYPE_NAME, Z_NTYPE_NAME, X_NTYPE_NAME, H_NTYPE_NAME]
ETYPE_NAMES = [D_ETYPE_NAME, S_ETYPE_NAME, H_ETYPE_NAME]

PHASE = 'phase'
DEGREE = 'degree'
NTYPE = 'type'
COLUMN = 'x'
ROW = 'y'
CONNECTED_TO = 'connected_to'
ETYPE = 'type'


def is_basis(ntype: str | int) -> bool:
    if isinstance(ntype, str) or isinstance(ntype, int):
        return is_z_basis(ntype) or is_x_basis(ntype)
    else:
        raise Exception('Unexpected node type representation ' + str(ntype))


def is_z_basis(ntype: str | int) -> bool:
    if isinstance(ntype, str):
        return ntype == Z_NTYPE_NAME
    elif isinstance(ntype, int):
        return ntype == Z_NTYPE_INDEX
    else:
        raise Exception('Unexpected node type representation ' + str(ntype))


def is_x_basis(ntype: str | int) -> bool:
    if isinstance(ntype, str):
        return ntype == X_NTYPE_NAME
    elif isinstance(ntype, int):
        return ntype == X_NTYPE_INDEX
    else:
        raise Exception('Unexpected node type representation ' + str(ntype))


def is_boundary(ntype: str | int) -> bool:
    if isinstance(ntype, str):
        return ntype == B_NTYPE_NAME
    elif isinstance(ntype, int):
        return ntype == B_NTYPE_INDEX
    else:
        raise Exception('Unexpected node type representation ' + str(ntype))


def is_hadamard_edge(etype: str | int) -> bool:
    if isinstance(etype, str):
        return etype == H_ETYPE_NAME
    elif isinstance(etype, int):
        return etype == H_ETYPE_INDEX
    else:
        raise Exception('Unexpected edge type representation ' + str(etype))


def is_simple_edge(etype: str | int) -> bool:
    return not is_hadamard_edge(etype)


PYG_ETYPE_NAMES = [
    (Z_NTYPE_NAME, H_ETYPE_NAME, Z_NTYPE_NAME),
    (Z_NTYPE_NAME, S_ETYPE_NAME, Z_NTYPE_NAME),
    (Z_NTYPE_NAME, H_ETYPE_NAME, X_NTYPE_NAME),
    (Z_NTYPE_NAME, S_ETYPE_NAME, X_NTYPE_NAME),
    (Z_NTYPE_NAME, H_ETYPE_NAME, B_NTYPE_NAME),
    (Z_NTYPE_NAME, S_ETYPE_NAME, B_NTYPE_NAME),
    (X_NTYPE_NAME, H_ETYPE_NAME, X_NTYPE_NAME),
    (X_NTYPE_NAME, S_ETYPE_NAME, X_NTYPE_NAME),
    (X_NTYPE_NAME, H_ETYPE_NAME, B_NTYPE_NAME),
    (X_NTYPE_NAME, S_ETYPE_NAME, B_NTYPE_NAME),
    (X_NTYPE_NAME, H_ETYPE_NAME, Z_NTYPE_NAME),
    (X_NTYPE_NAME, S_ETYPE_NAME, Z_NTYPE_NAME),
    (B_NTYPE_NAME, H_ETYPE_NAME, B_NTYPE_NAME),
    (B_NTYPE_NAME, S_ETYPE_NAME, B_NTYPE_NAME),
    (B_NTYPE_NAME, H_ETYPE_NAME, Z_NTYPE_NAME),
    (B_NTYPE_NAME, S_ETYPE_NAME, Z_NTYPE_NAME),
    (B_NTYPE_NAME, H_ETYPE_NAME, X_NTYPE_NAME),
    (B_NTYPE_NAME, S_ETYPE_NAME, X_NTYPE_NAME)
]

PYG_ETYPE_NAMES_TO_INDICES = {name: i for i, name in enumerate(PYG_ETYPE_NAMES)}


def edge_type_index(nx_graph: nx.Graph, u: int, v: int, etype: int) -> int:
    return PYG_ETYPE_NAMES_TO_INDICES[
        (NTYPE_NAMES[nx_graph.nodes[u][NTYPE]], ETYPE_NAMES[etype],
         NTYPE_NAMES[nx_graph.nodes[v][NTYPE]])]


def node_types(nx_graph: nx.Graph) -> torch.Tensor:
    return torch.tensor([t for _, t in nx_graph.nodes(data=NTYPE)])


def edge_types(nx_graph: nx.Graph) -> torch.Tensor:
    return torch.tensor([edge_type_index(nx_graph, u, v, t) for u, v, t in nx_graph.edges(data=NTYPE)])


def nx_remove_position_attributes(nx_graph: nx.MultiGraph) -> None:
    for _, ndata in nx_graph.nodes(data=True):
        del ndata[ROW]
        del ndata[COLUMN]


def nx_remove_top_level_attributes(nx_graph: nx.MultiGraph) -> None:
    del nx_graph.graph['node_default']
    del nx_graph.graph['edge_default']


def nx_add_boundary_connected_to(nx_graph: nx.MultiGraph) -> None:
    for n, ndata in nx_graph.nodes(data=True):
        if is_boundary(ndata[NTYPE]):
            ndata[CONNECTED_TO] = nx_graph.nodes[list(nx_graph.neighbors(n))[0]][NTYPE]


def nx_add_degree(nx_graph: nx.MultiGraph) -> None:
    for n, ndata in nx_graph.nodes(data=True):
        ndata[DEGREE] = len(list(nx_graph.neighbors(n)))


def nx_remove_boundary_phase(nx_graph: nx.MultiGraph) -> None:
    for n, ndata in nx_graph.nodes(data=True):
        if is_boundary(ndata[NTYPE]):
            del ndata[PHASE]


def nx_to_pyg_heterograph_pre_process(nx_graph: nx.MultiGraph) -> None:
    nx_remove_position_attributes(nx_graph)
    nx_remove_top_level_attributes(nx_graph)
    nx_add_boundary_connected_to(nx_graph)
    nx_add_degree(nx_graph)
    nx_remove_boundary_phase(nx_graph)


def node_type_to_one_hot(hdata: pyg.data.HeteroData) -> None:
    pass


def edge_type_to_one_hot(hdata: pyg.data.HeteroData) -> None:
    pass


def nx_to_pyg_heterograph_post_process(hdata: pyg.data.HeteroData) -> None:
    node_type_to_one_hot(hdata)
    edge_type_to_one_hot(hdata)


def nx_to_pyg_hetero(
    G: Any,
    node_type_attribute: str,
    edge_type_attribute: Optional[str] = None,
    graph_attrs: Optional[Iterable[str]] = None, nodes: Optional[List] = None,
    group_node_attrs: Optional[Union[List[str], all]] = None,
    group_edge_attrs: Optional[Union[List[str], all]] = None
) -> pyg.data.HeteroData:

    def get_edge_attributes(G, edge_indexes: list,
                            edge_attrs: list = None) -> dict:
        r"""Collects the attributes of a list of graph edges in a dictionary.

        Args:
            G (networkx.Graph or networkx.DiGraph): A networkx graph.
            edge_indexes (list, optional): The list of edge indexes whose
                attributes are to be collected. If set to :obj:`None`, all
                edges of the graph will be included. (default: :obj:`None`)
            edge_attrs (list, optional): The list of expected attributes to
                be found in every edge. If set to :obj:`None`, the first
                edge encountered will set the values for the rest of the
                process. (default: :obj:`None`)

        Raises:
            ValueError: If some of the edges do not share the same list
            of attributes as the rest, an error will be raised.
        """
        data = defaultdict(list)
        edge_to_data = list(G.edges(data=True))

        for edge_index in edge_indexes:
            _, _, feat_dict = edge_to_data[edge_index]
            if edge_attrs is None:
                edge_attrs = feat_dict.keys()
            if set(feat_dict.keys()) != set(edge_attrs):
                raise ValueError('Not all edges contain the same attributes.')
            for key, value in feat_dict.items():
                data[str(key)].append(value)

        return data

    def get_node_attributes(G, nodes: list,
                            expected_node_attrs: list = None) -> dict:
        r"""Collects the attributes of a list of graph nodes in a dictionary.

        Args:
            G (networkx.Graph or networkx.DiGraph): A networkx graph.
            nodes (list, optional): The list of nodes whose attributes are to
                be collected. If set to :obj:`None`, all nodes of the graph
                will be included. (default: :obj:`None`)
            expected_node_attrs (list, optional): The list of expected
                attributes to be found in every node. If set to :obj:`None`,
                the first node encountered will set the values for the rest
                of the process. (default: :obj:`None`)

        Raises:
            ValueError: If some of the nodes do not share the same
            list of attributes as the rest, an error will be raised.
        """

        data = defaultdict(list)

        node_to_data = G.nodes(data=True)

        for node in nodes:
            feat_dict = node_to_data[node]
            if expected_node_attrs is None:
                expected_node_attrs = feat_dict.keys()
            if set(feat_dict.keys()) != set(expected_node_attrs):
                raise ValueError('Not all nodes contain the same attributes.')
            for key, value in feat_dict.items():
                data[str(key)].append(value)

        return data

    G = G.to_directed() if not nx.is_directed(G) else G

    if nodes is not None:
        G = nx.subgraph(G, nodes)

    hetero_data_dict = {}

    node_to_group_id = {}
    node_to_group = {}
    group_to_nodes = defaultdict(list)
    group_to_edges = defaultdict(list)

    for node, node_data in G.nodes(data=True):
        if node_type_attribute not in node_data:
            raise KeyError(f"Given node_type_attribute: {node_type_attribute} \
                missing from node {node}.")
        node_type = str(node_data[node_type_attribute])
        group_to_nodes[node_type].append(node)
        node_to_group_id[node] = len(group_to_nodes[node_type]) - 1
        node_to_group[node] = node_type

    for i, (node_a, node_b, edge_data) in enumerate(G.edges(data=True)):
        if edge_type_attribute is not None:
            if edge_type_attribute not in edge_data:
                raise KeyError(
                    f"Given edge_type_attribute: {edge_type_attribute} \
                    missing from edge {(node_a, node_b)}.")
            print('edata = ', edge_data[
                edge_type_attribute])
            node_type_a, edge_type, node_type_b = edge_data[
                edge_type_attribute]
            if node_to_group[node_a] != node_type_a or node_to_group[
                    node_b] != node_type_b:
                raise ValueError(f'Edge {node_a}-{node_b} of type\
                         {edge_data[edge_type_attribute]} joins nodes of types\
                         {node_to_group[node_a]} and { node_to_group[node_b]}.'
                                 )
        else:
            edge_type = "to"
        group_to_edges[(node_to_group[node_a], edge_type,
                        node_to_group[node_b])].append(i)

    for group, group_nodes in group_to_nodes.items():
        hetero_data_dict[str(group)] = {
            k: v
            for k, v in get_node_attributes(G, nodes=group_nodes).items()
            if k != node_type_attribute
        }

    for group, group_edges in group_to_edges.items():
        group_name = '__'.join(group)
        hetero_data_dict[group_name] = {
            k: v
            for k, v in get_edge_attributes(G, edge_indexes=group_edges).items()
            if k != edge_type_attribute
        }
        edge_list = list(G.edges(data=False))
        global_edge_index = [edge_list[edge] for edge in group_edges]
        group_edge_index = [(node_to_group_id[node_a],
                             node_to_group_id[node_b])
                            for node_a, node_b in global_edge_index]
        hetero_data_dict[group_name]['edge_index'] = torch.tensor(
            group_edge_index, dtype=torch.long).t().contiguous().view(2, -1)

    graph_items = G.graph
    if graph_attrs is not None:
        graph_items = {
            k: v
            for k, v in graph_items.items() if k in graph_attrs
        }
    for key, value in graph_items.items():
        hetero_data_dict[str(key)] = value

    for group, group_dict in hetero_data_dict.items():
        if isinstance(group_dict, dict):
            xs = []
            is_edge_group = group in [
                '__'.join(k) for k in group_to_edges.keys()
            ]
            if is_edge_group:
                group_attrs = group_edge_attrs
            else:
                group_attrs = group_node_attrs
            for key, value in group_dict.items():
                if isinstance(value, (tuple, list)) and isinstance(
                        value[0], torch.Tensor):
                    hetero_data_dict[group][key] = torch.stack(value, dim=0)
                else:
                    try:
                        hetero_data_dict[group][key] = torch.tensor(value)
                    except (ValueError, TypeError):
                        pass
                if group_attrs is not None and key in group_attrs:
                    xs.append(hetero_data_dict[group][key].view(-1, 1))
            if group_attrs is not None:
                if len(group_attrs) != len(xs):
                    raise KeyError(
                        f'Missing required attribute in group: {group}')
                if is_edge_group:
                    hetero_data_dict[group]['edge_attr'] = torch.cat(
                        xs, dim=-1)
                else:
                    hetero_data_dict[group]['x'] = torch.cat(xs, dim=-1)
        else:
            value = group_dict
            if isinstance(value, (tuple, list)) and isinstance(
                    value[0], torch.Tensor):
                hetero_data_dict[group] = torch.stack(value, dim=0)
            else:
                try:
                    hetero_data_dict[group] = torch.tensor(value)
                except (ValueError, TypeError):
                    pass

    return HeteroData(**hetero_data_dict)