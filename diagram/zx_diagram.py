from collections.abc import Iterable

import networkx as nx

from diagram.pyzx_nx_conv import is_basis, is_boundary, is_z_basis, is_x_basis, X_NTYPE_INDEX, \
    Z_NTYPE_INDEX, S_ETYPE_INDEX, B_NTYPE_INDEX


class ZXDiagram(nx.MultiGraph):
    NTYPE = 'type'
    PHASE = 'phase'
    ETYPE = 'type'

    def __init__(self, nx_graph: nx.MultiGraph = None):
        # Assuming current nodes are 0-based consecutive integers...
        if nx_graph is None:
            nx_graph = nx.MultiGraph()
        self.next_node_index = max(nx_graph.nodes(data=False)) + 1 if nx_graph.number_of_nodes() > 0 else 0
        self.z_nodes_set = set()
        self.x_nodes_set = set()
        self.b_nodes_set = set()
        super().__init__(incoming_graph_data=nx_graph, multigraph_input=True)
        for n in nx_graph.nodes:
            if self.is_x_basis(n):
                self.x_nodes_set.add(n)
            elif self.is_z_basis(n):
                self.z_nodes_set.add(n)
            elif self.is_boundary(n):
                self.b_nodes_set.add(n)
            else:
                raise Exception(f'Node {n} has undefined type')

    @property
    def node_attrs(self):
        return [self.NTYPE, self.PHASE]

    @property
    def edge_attrs(self):
        return [self.ETYPE]

    def type(self, n: int) -> int:
        assert self.has_node(n), f'Node {n} does not exist'
        return self.nodes[n][self.NTYPE]

    def types(self) -> dict[int, int]:
        return {n: ndata[self.NTYPE] for n, ndata in self.nodes(data=True)}

    def phase(self, n: int) -> float:
        assert self.has_node(n), f'Node {n} does not exist'
        return self.nodes[n][self.PHASE]

    def phases(self) -> dict[int, float]:
        return {n: ndata[self.PHASE] for n, ndata in self.nodes(data=True)}

    def set_phase(self, n: int, phase: float) -> None:
        assert self.is_basis(n), f'Attempted to set phase of non-basis node {n}'
        self.nodes[n][self.PHASE] = phase

    def is_boundary(self, n: int) -> bool:
        assert self.has_node(n), f'Node {n} does not exist'
        return is_boundary(self.nodes[n][self.NTYPE])

    def is_basis(self, n: int) -> bool:
        assert self.has_node(n), f'Node {n} does not exist'
        return is_basis(self.nodes[n][self.NTYPE])

    def basis_nodes(self) -> set[int]:
        return self.x_nodes().union(self.z_nodes())

    def is_z_basis(self, n: int) -> bool:
        assert self.has_node(n), f'Node {n} does not exist'
        return is_z_basis(self.nodes[n][self.NTYPE])

    def is_x_basis(self, n: int) -> bool:
        assert self.has_node(n), f'Node {n} does not exist'
        return is_x_basis(self.nodes[n][self.NTYPE])

    def flip_basis(self, n: int) -> None:
        assert self.is_basis(n), f'Attempted to basis flip non-basis node {n}'
        (self.x_nodes_set.remove if self.is_x_basis(n) else self.z_nodes_set.remove)(n)
        self.nodes[n][self.NTYPE] = (X_NTYPE_INDEX if self.is_z_basis(n) else Z_NTYPE_INDEX)
        (self.x_nodes_set.add if self.is_x_basis(n) else self.z_nodes_set.add)(n)

    def add_x_node(self, phase: float) -> int:
        assert -2 < phase < 2, f'Attempted to add Z-basis node with invalid phase {phase}'
        new_x = self.__next_node()
        self.add_node(new_x, type=X_NTYPE_INDEX, phase=phase)
        self.x_nodes_set.add(new_x)
        return new_x

    def add_x_nodes(self, phases: list[float]) -> list[int]:
        return [self.add_x_node(phase) for phase in phases]

    def remove_nodes_from(self, nodes):
        for n in nodes:
            if self.is_x_basis(n):
                self.remove_x_node(n)
            elif self.is_z_basis(n):
                self.remove_z_node(n)
            elif self.is_boundary(n):
                self.remove_b_node(n)
            else:
                raise Exception(f'Node {n} has undefined type')

    def x_nodes(self) -> set[int]:
        return self.x_nodes_set

    def num_x_nodes(self) -> int:
        return len(self.x_nodes())

    def remove_x_node(self, n: int) -> None:
        assert self.is_x_basis(n), f'Attempted to remove non-X-basis node {n}'
        self.remove_node(n)
        self.x_nodes_set.remove(n)

    def add_z_node(self, phase: float) -> int:
        assert -2 < phase < 2, f'Attempted to add Z-basis node with invalid phase {phase}'
        new_z = self.__next_node()
        self.add_node(new_z, type=Z_NTYPE_INDEX, phase=phase)
        self.z_nodes_set.add(new_z)
        return new_z

    def add_z_nodes(self, phases: list[float]) -> list[int]:
        return [self.add_z_node(phase) for phase in phases]

    def z_nodes(self) -> set[int]:
        return self.z_nodes_set

    def num_z_nodes(self) -> int:
        return len(self.z_nodes())

    def remove_z_node(self, n: int) -> None:
        assert self.is_z_basis(n), f'Attempted to remove non-Z-basis node {n}'
        self.remove_node(n)
        self.z_nodes_set.remove(n)

    def add_b_node(self) -> int:
        new_b = self.__next_node()
        self.add_node(new_b, type=B_NTYPE_INDEX, phase=0)
        self.b_nodes_set.add(new_b)
        return new_b

    def add_b_nodes(self, count: int) -> list[int]:
        return [self.add_b_node() for _ in range(count)]

    def b_nodes(self) -> set[int]:
        return self.b_nodes_set

    def num_b_nodes(self) -> int:
        return len(self.b_nodes())

    def remove_b_node(self, n: int) -> None:
        assert self.is_boundary(n), f'Attempted to remove non-boundary node {n}'
        self.remove_node(n)
        self.b_nodes_set.remove(n)

    def add_s_edge(self, s: int, t: int) -> int:
        assert self.has_node(s), f'Node {s} does not exist'
        assert self.has_node(s), f'Node {t} does not exist'
        return self.add_edge(s, t, type=S_ETYPE_INDEX)

    def add_s_edges_from(self, es: Iterable[tuple[int, int]]) -> list[int]:
        for s, t in es:
            assert self.has_node(s), f'Node {s} does not exist'
            assert self.has_node(t), f'Node {t} does not exist'
        return self.add_edges_from(es, type=S_ETYPE_INDEX)

    def remove_edges(self, s: int, t: int) -> None:
        assert self.has_node(s), f'Node {s} does not exist'
        assert self.has_node(t), f'Node {t} does not exist'
        self.remove_edges_from(self.edges_between(s, t))

    def incident_edges(self, n: int) -> set[tuple[int, int, int]]:
        assert self.has_node(n), f'Node {n} does not exist'
        edges = set()
        for neighbor in self.neighbors(n):
            # Relies on the fact that edges in this iterator are given with respect to the order of the
            # nodes in the 'nbunch' argument, i.e., n always comes first.
            for _, _, key in self.edges_between(n, neighbor):
                edges.add((n, neighbor, key))
        return edges

    def remove_incident_edges(self, n: int) -> None:
        assert self.has_node(n), f'Node {n} does not exist'
        self.remove_edges_from(self.incident_edges(n))

    def neighbors_from(self, ns: Iterable[int]) -> set[int]:
        neighbors = set()
        for n in ns:
            assert self.has_node(n), f'Node {n} does not exist'
            neighbors.update(self.neighbors(n))
        return neighbors

    def __next_node(self) -> int:
        next_node_index = self.next_node_index
        while self.has_node(next_node_index):
            next_node_index = next_node_index + 1
        return next_node_index

    def edges_between(self, n: int, m: int, data=False) -> list[
            tuple[int, int, int] | tuple[int, int, int, dict[str, any]]]:
        assert self.has_node(n), f'Node {n} does not exist'
        assert self.has_node(n), f'Node {m} does not exist'
        if data:
            return [(n, m, k, edata) for k, edata in self[n][m].items()]
        return [(n, m, k) for k in self[n][m]]
