from collections.abc import Iterable
from typing import Iterator

import networkx as nx
import torch_geometric as pyg

from diagram.match import Match, FRightMatch, FRightZMatch, FRightXMatch, FLeftZMatch, FLeftMatch, FLeftXMatch, \
    BLeftMatch, BRightMatch, YLeftMatch, YRightMatch, YLeftXMatch, YLeftZMatch, YRightZMatch, YRightXMatch
from diagram.pyzx_nx_conv import is_basis, is_boundary, is_z_basis, is_x_basis, \
    Z_NTYPE_NAME, B_NTYPE_NAME, X_NTYPE_NAME
from diagram.zx_match_diagram import ZXMatchDiagram


class ZXDiagram(nx.MultiGraph):
    NTYPE = 'type'
    PHASE = 'phase'

    def __init__(self, phase_denominator: int, nx_graph: nx.MultiGraph = None):
        if phase_denominator <= 0:
            raise ValueError(f"The phase denominator {phase_denominator} must be positive.")
        self.phase_denominator = phase_denominator
        super().__init__(incoming_graph_data=nx_graph, multigraph_input=True)
        self.next_node_index = max(nx_graph.nodes(data=False)) + 1 if self.number_of_nodes() > 0 else 0
        self.__initialize_graph_from_nx_graph(nx_graph)

    def __initialize_graph_from_nx_graph(self, nx_graph: nx.MultiGraph = None):
        if nx_graph is not None:
            for n in nx_graph.nodes:
                self.__validate_and_add_phase(n, nx_graph.nodes[n][self.PHASE])
                print(nx_graph.nodes[n])
            self._z_nodes_set = set()
            self._x_nodes_set = set()
            self._b_nodes_set = set()
            for n in nx_graph.nodes:
                if self.is_x_basis(n):
                    self._x_nodes_set.add(n)
                elif self.is_z_basis(n):
                    self._z_nodes_set.add(n)
                elif self.is_boundary(n):
                    self._b_nodes_set.add(n)
                else:
                    raise Exception(f'Node {n} has unexpected type')

    def __validate_and_add_phase(self, n: int, phase: float):
        # Validate phase
        if not self.is_valid_phase(phase):
            raise ValueError(
                f'Phase {phase} for node {n} is invalid for diagram with phase denominator {self.phase_denominator}')
        # Set phase
        self.nodes[n][self.PHASE] = phase

    def is_valid_phase(self, phase: float) -> bool:
        normalized_position = (phase * self.phase_denominator) % self.phase_denominator
        return normalized_position.is_integer()

    @property
    def node_attrs(self):
        return [self.NTYPE, self.PHASE]

    @property
    def edge_attrs(self):
        return []

    def type(self, n: int) -> str:
        assert self.has_node(n), f'Node {n} does not exist'
        return self.nodes[n][self.NTYPE]

    def types(self) -> dict[int, str]:
        return {n: ndata[self.NTYPE] for n, ndata in self.nodes(data=True)}

    def phase(self, n: int) -> float:
        assert self.is_basis(n), f'Node {n} is a boundary node'
        return self.nodes[n][self.PHASE]

    def phases(self) -> dict[int, float]:
        return {n: self.nodes[n][self.PHASE] for n in self.basis_nodes()}

    def set_phase(self, n: int, phase: float) -> None:
        assert self.is_basis(n), f'Attempted to set phase of non-basis node {n}'
        assert self.is_valid_phase(
            phase), f'Phase {phase} is invalid for diagram with phase denominator {self.phase_denominator}'
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
        (self._x_nodes_set.remove if self.is_x_basis(n) else self._z_nodes_set.remove)(n)
        self.nodes[n][self.NTYPE] = (X_NTYPE_NAME if self.is_z_basis(n) else Z_NTYPE_NAME)
        (self._x_nodes_set.add if self.is_x_basis(n) else self._z_nodes_set.add)(n)

    def add_x_node(self, phase: float) -> int:
        assert self.is_valid_phase(
            phase), f'Phase {phase} for X-basis node is invalid for diagram with phase denominator {self.phase_denominator}'
        new_x = self.__next_node()
        self.add_node(new_x, type=X_NTYPE_NAME, phase=phase)
        self._x_nodes_set.add(new_x)
        return new_x

    def add_x_nodes(self, phases: list[float]) -> list[int]:
        return [self.add_x_node(phase) for phase in phases]

    def remove_nodes_from(self, nodes: list[int]) -> None:
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
        return self._x_nodes_set.copy()

    def num_x_nodes(self) -> int:
        return len(self.x_nodes())

    def remove_x_node(self, n: int) -> None:
        assert self.is_x_basis(n), f'Attempted to remove non-X-basis node {n}'
        self.remove_node(n)
        self._x_nodes_set.remove(n)

    def add_z_node(self, phase: float) -> int:
        assert self.is_valid_phase(
            phase), f'Phase {phase} for Z-basis node is invalid for diagram with phase denominator {self.phase_denominator}'
        new_z = self.__next_node()
        self.add_node(new_z, type=Z_NTYPE_NAME, phase=phase)
        self._z_nodes_set.add(new_z)
        return new_z

    def add_z_nodes(self, phases: list[float]) -> list[int]:
        return [self.add_z_node(phase) for phase in phases]

    def z_nodes(self) -> set[int]:
        return self._z_nodes_set.copy()

    def num_z_nodes(self) -> int:
        return len(self.z_nodes())

    def remove_z_node(self, n: int) -> None:
        assert self.is_z_basis(n), f'Attempted to remove non-Z-basis node {n}'
        self.remove_node(n)
        self._z_nodes_set.remove(n)

    def add_b_node(self) -> int:
        new_b = self.__next_node()
        self.add_node(new_b, type=B_NTYPE_NAME, phase=0)
        self._b_nodes_set.add(new_b)
        return new_b

    def add_b_nodes(self, count: int) -> list[int]:
        return [self.add_b_node() for _ in range(count)]

    def b_nodes(self) -> set[int]:
        return self._b_nodes_set.copy()

    def num_b_nodes(self) -> int:
        return len(self.b_nodes())

    def remove_b_node(self, n: int) -> None:
        assert self.is_boundary(n), f'Attempted to remove non-boundary node {n}'
        self.remove_node(n)
        self._b_nodes_set.remove(n)

    def add_s_edge(self, s: int, t: int) -> int:
        assert self.has_node(s), f'Node {s} does not exist'
        assert self.has_node(s), f'Node {t} does not exist'
        return self.add_edge(s, t)

    def add_s_edges_from(self, es: Iterable[tuple[int, int]]) -> list[int]:
        for s, t in es:
            assert self.has_node(s), f'Node {s} does not exist'
            assert self.has_node(t), f'Node {t} does not exist'
        return self.add_edges_from(es)

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

    def f_left_z_matches(self) -> Iterator[FLeftZMatch]:
        candidates = set()
        for s, t, edata in self.edges(data=True):
            if self.is_z_basis(s) and self.is_z_basis(t):
                candidates.add((s, t))
        for s, t in candidates:
            yield FLeftZMatch({s: 0, t: 1})

    def f_left_x_matches(self) -> Iterator[FLeftXMatch]:
        candidates = set()
        for s, t, edata in self.edges(data=True):
            if self.is_x_basis(s) and self.is_x_basis(t):
                candidates.add((s, t))
        for s, t in candidates:
            yield FLeftXMatch({s: 0, t: 1})

    def f_left_matches(self) -> Iterator[FLeftMatch]:
        yield from self.f_left_z_matches()
        yield from self.f_left_x_matches()

    def f_right_z_matches(self) -> Iterator[FRightZMatch]:
        yield from (FRightZMatch({z: 0}) for z in self.z_nodes())

    def f_right_x_matches(self) -> Iterator[FRightXMatch]:
        yield from (FRightXMatch({x: 0}) for x in self.x_nodes())

    def f_right_matches(self) -> Iterator[FRightMatch]:
        yield from self.f_right_z_matches()
        yield from self.f_right_x_matches()

    def b_left_matches(self) -> Iterator[BLeftMatch]:
        candidates = {(s, t) if self.is_z_basis(s) and self.is_x_basis(t) else (t, s) for s, t in
                      set(self.edges(data=False))
                      if self.degree(s) == 3 and self.degree(t) == 3 and
                      self.is_basis(s) and self.is_basis(t) and self.type(s) != self.type(t) and
                      self.phase(s) == 0 and self.phase(t) == 0}
        while len(candidates) > 0:
            z, x = candidates.pop()
            for n in self.neighbors(z):
                if not n == x and self.is_x_basis(n) and self.phase(n) == 0 and self.degree(n) == 3:
                    for m in self.neighbors(n):
                        if not m == z and self.is_z_basis(m) and self.phase(m) == 0 and self.degree(m) == 3:
                            for o in self.neighbors(m):
                                if o == x:
                                    candidates.discard((z, x))
                                    candidates.discard((z, n))
                                    candidates.discard((m, x))
                                    candidates.discard((m, n))
                                    yield BLeftMatch(z, x, m, n)

    def b_right_matches(self) -> Iterator[BRightMatch]:
        for s, t in set(self.edges(data=False)):
            if self.degree(s) == 3 and self.degree(t) == 3:
                if self.is_basis(s) and self.is_basis(t) and self.type(s) != self.type(t):
                    if self.phase(s) == 0 and self.phase(t) == 0:
                        x, z = (s, t) if self.is_x_basis(s) and self.is_z_basis(t) else (t, s)
                        yield BRightMatch(x, z)

    def y_left_z_matches(self) -> Iterator[YLeftZMatch]:
        for n in self.x_nodes():
            if self.degree(n) == 3 and self.phase(n) == 0:
                if all([self.degree(m) == 2 and self.is_z_basis(m) for m in self.neighbors(n)]) and sum(
                        [self.phase(m) for m in self.neighbors(n)]) == 0.5:
                    z0, z2, z3 = sorted(self.neighbors(n), key=lambda m: self.phase(m))
                    yield YLeftZMatch(z0, n, z2, z3)

    def y_left_x_matches(self) -> Iterator[YLeftXMatch]:
        for n in self.z_nodes():
            if self.degree(n) == 3 and self.phase(n) == 0:
                if all([self.degree(m) == 2 and self.is_x_basis(m) for m in self.neighbors(n)]) and sum(
                        [self.phase(m) for m in self.neighbors(n)]) == 0.5:
                    x0, x2, x3 = sorted(self.neighbors(n), key=lambda m: self.phase(m))
                    yield YLeftXMatch(x0, n, x2, x3)

    def y_left_matches(self) -> Iterator[YLeftMatch]:
        yield from self.y_left_z_matches()
        yield from self.y_left_x_matches()

    def y_right_z_matches(self) -> Iterator[YRightZMatch]:
        for n in self.x_nodes():
            if self.degree(n) == 3 and self.phase(n) == -0.5:
                if all([self.degree(m) == 2 and self.is_z_basis(m) for m in self.neighbors(n)]) and sum(
                        [self.phase(m) for m in self.neighbors(n)]) == -0.5:
                    x0, x2, x3 = sorted(self.neighbors(n), key=lambda m: self.phase(m), reverse=True)
                    yield YRightZMatch(x0, n, x2, x3)

    def y_right_x_matches(self) -> Iterator[YRightXMatch]:
        for n in self.z_nodes():
            if self.degree(n) == 3 and self.phase(n) == -0.5:
                if all([self.degree(m) == 2 and self.is_x_basis(m) for m in self.neighbors(n)]) and sum(
                        [self.phase(m) for m in self.neighbors(n)]) == -0.5:
                    x0, x2, x3 = sorted(self.neighbors(n), key=lambda m: self.phase(m), reverse=True)
                    yield YRightXMatch(x0, n, x2, x3)

    def y_right_matches(self) -> Iterator[YRightMatch]:
        yield from self.y_right_z_matches()
        yield from self.y_right_x_matches()

    def compute_matches(self) -> Iterator[Match]:
        yield from self.f_left_matches()
        yield from self.f_right_matches()
        yield from self.b_left_matches()
        yield from self.b_right_matches()
        yield from self.y_left_matches()
        yield from self.y_right_matches()

    def to_zx_match_diagram(self) -> ZXMatchDiagram:
        pass

    def to_pyg_data(self, one_hot_types=True) -> pyg.data.Data:
        pass

    def to_pyg_hetero_data(self, one_hot_types=True) -> pyg.data.HeteroData:
        pass

    def copy(self, as_view=False):
        diagram_copy = self.__class__(self.phase_denominator, self)
        diagram_copy.add_nodes_from(self.nodes)
        diagram_copy.add_edges_from(self.edges)
        return diagram_copy
