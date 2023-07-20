from collections.abc import Iterable

import networkx as nx

from graph.pyzx_nx_conv import is_basis, is_boundary, is_z_basis, is_x_basis, X_NTYPE_INDEX, \
    Z_NTYPE_INDEX, S_ETYPE_INDEX, B_NTYPE_INDEX


class ZXDiagram(nx.MultiGraph):

    NTYPE = 'type'
    PHASE = 'phase'
    DEGREE = 'degree'
    ETYPE = 'type'

    def __init__(self, nx_graph: nx.MultiGraph = None):
        # Assuming current nodes are 0-based consecutive integers...
        if nx_graph is None:
            nx_graph = nx.MultiGraph()
        self.next_node_index = nx_graph.number_of_nodes()
        super().__init__(incoming_graph_data=nx_graph, multigraph_input=True)
        for n, ndata in nx_graph.nodes(data=True):
            ndata[self.DEGREE] = self.degree(n)

    @property
    def node_attrs(self):
        return [self.NTYPE, self.PHASE]

    @property
    def edge_attrs(self):
        return [self.ETYPE]

    def get_phase(self, n: int) -> float:
        assert self.has_node(n), f'Node {n} does not exist'
        return self.nodes[n][self.PHASE]

    def get_degree(self, n: int) -> int:
        return self.nodes[n][self.DEGREE]

    def set_phase(self, n: int, phase: float) -> None:
        assert self.is_basis(n), f'Attempted to set phase of non-basis node {n}'
        self.nodes[n][self.PHASE] = phase

    def is_boundary(self, n: int) -> bool:
        assert self.has_node(n), f'Node {n} does not exist'
        return is_boundary(self.nodes[n][self.NTYPE])

    def is_basis(self, n: int) -> bool:
        assert self.has_node(n), f'Node {n} does not exist'
        return is_basis(self.nodes[n][self.NTYPE])

    def is_z_basis(self, n: int) -> bool:
        assert self.has_node(n), f'Node {n} does not exist'
        return is_z_basis(self.nodes[n][self.NTYPE])

    def is_x_basis(self, n: int) -> bool:
        assert self.has_node(n), f'Node {n} does not exist'
        return is_x_basis(self.nodes[n][self.NTYPE])

    def flip_basis(self, n: int) -> None:
        assert self.is_basis(n), f'Attempted to basis flip non-basis node {n}'
        self.nodes[n][self.PHASE] = Z_NTYPE_INDEX if self.is_x_basis(n) else X_NTYPE_INDEX

    def add_x_node(self, phase: float) -> int:
        new_x = self.__next_node()
        self.add_node(new_x, type=X_NTYPE_INDEX, phase=phase, degree=0)
        return new_x

    def remove_x_node(self, n: int) -> None:
        assert self.is_x_basis(n), f'Attempted to remove non-X-basis node {n}'
        self.remove_node(n)

    def add_z_node(self, phase: float) -> int:
        new_z = self.__next_node()
        self.add_node(new_z, type=Z_NTYPE_INDEX, phase=phase, degree=0)
        return new_z

    def remove_z_node(self, n: int) -> None:
        assert self.is_z_basis(n), f'Attempted to remove non-Z-basis node {n}'
        self.remove_node(n)

    def add_b_node(self) -> int:
        new_b = self.__next_node()
        self.add_node(new_b, type=B_NTYPE_INDEX, phase=0, degree=0)
        return new_b

    def remove_b_node(self, n: int) -> None:
        assert self.is_boundary(n), f'Attempted to remove non-boundary node {n}'
        self.remove_node(n)

    def add_s_edge(self, s: int, t: int) -> int:
        key = self.add_edge(s, t, type=S_ETYPE_INDEX)
        self.__set_degree(s, self.degree(s))
        self.__set_degree(t, self.degree(t))
        return key

    def add_s_edges_from(self, es: Iterable[tuple[int, int]]) -> list[int]:
        keys = self.add_edges_from(es, type=S_ETYPE_INDEX)
        for s, t in es:
            self.__set_degree(s, self.degree(s))
            self.__set_degree(t, self.degree(t))
        return keys

    def remove_edges(self, s: int, t: int) -> None:
        assert self.has_node(s), f'Node {s} does not exist'
        assert self.has_node(t), f'Node {t} does not exist'
        self.remove_edges_from(self.__edges_between(s, t))
        self.__set_degree(s, self.degree(s))
        self.__set_degree(t, self.degree(t))

    def incident_edges(self, n: int) -> set[tuple[int, int, int]]:
        edges = set()
        for neighbor in self.neighbors(n):
            # Relies on the fact that edges in this iterator are given with respect to the order of the
            # nodes in the 'nbunch' argument, i.e., n always comes first.
            for _, _, key in self.__edges_between(n, neighbor):
                edges.add((n, neighbor, key))
        return edges

    def remove_incident_edges(self, n: int) -> None:
        self.remove_edges_from(self.incident_edges(n))
        self.__set_degree(n, self.degree(n))
        for neighbor in self.neighbors(n):
            self.__set_degree(neighbor, self.degree(neighbor))

    def neighbors_from(self, ns: Iterable[int]) -> set[int]:
        neighbors = set()
        for n in ns:
            for neighborhood in self.neighbors(n):
                neighbors.update(neighborhood)
        return neighbors

    def __next_node(self) -> int:
        next_node_index = self.next_node_index
        self.next_node_index = self.next_node_index + 1
        if self.has_node(next_node_index):
            raise Exception(f'Bug found: expected node index {next_node_index} to be unused')
        return next_node_index

    def __set_degree(self, n: int, degree: int) -> None:
        self.nodes[n][self.DEGREE] = degree

    def __edges_between(self, n: int, m: int) -> set[tuple[int, int, int]]:
        return set([(n, m, k) for k in self[n][m].keys()])
