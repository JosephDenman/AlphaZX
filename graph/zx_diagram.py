import torch
from enum import Enum
from typing import Optional, Tuple


class NodeType(Enum):
    Z = 'z'
    X = 'x'
    BOUNDARY = 'boundary'


class EdgeType(Enum):
    SIMPLE = 'simple'
    HADAMARD = 'hadamard'


def canonical_edge_type(utype: NodeType, vtype: NodeType, etype: EdgeType) -> Tuple[str, str, str]:
    return utype.value, etype.value, vtype.value


def node_is_zx(node_type: NodeType) -> bool:
    """Check if a vertex type corresponds to a green or red spider."""
    return node_type in [NodeType.Z, NodeType.X]


class ZXDiagram(object):
    def __init__(self):
        pass

    def add_node(self, node_type: NodeType, phase: Optional[float] = None) -> None:
        pass

    def add_z_node(self, phase: float) -> None:
        pass

    def add_x_node(self, phase: float) -> None:
        pass

    def add_boundary_node(self) -> None:
        pass

    def z_nodes(self) -> int:
        pass

    def x_nodes(self) -> int:
        pass

    def boundary_nodes(self) -> int:
        pass

    def z_phases(self) -> torch.Tensor:
        pass

    def x_phases(self) -> torch.Tensor:
        pass

    def num_z_nodes(self) -> int:
        pass

    def num_x_nodes(self) -> int:
        pass

    def num_boundary_nodes(self) -> int:
        pass

    def num_nodes(self) -> int:
        pass

    def add_edge(self, u: int, utype: NodeType, v: int, vtype: NodeType, etype: EdgeType) -> None:
        pass

    def add_z_z_simple_edge(self, u: int, v: int) -> None:
        pass

    def add_z_z_hadamard_edge(self, u: int, v: int) -> None:
        pass

    def add_z_x_simple_edge(self, u: int, v: int) -> None:
        pass

    def add_z_x_hadamard_edge(self, u: int, v: int) -> None:
        pass

    def add_z_boundary_simple_edge(self, u: int, v: int) -> None:
        pass

    def add_z_boundary_hadamard_edge(self, u: int, v: int) -> None:
        pass

    def add_x_x_simple_edge(self, u: int, v: int) -> None:
        pass

    def add_x_x_hadamard_edge(self, u: int, v: int) -> None:
        pass

    def add_x_boundary_simple_edge(self, u: int, v: int) -> None:
        pass

    def add_x_boundary_hadamard_edge(self, u: int, v: int) -> None:
        pass

    def add_boundary_boundary_simple_edge(self, u: int, v: int) -> None:
        pass

    def add_boundary_boundary_hadamard_edge(self, u: int, v: int) -> None:
        pass

    def edges(self, utype: NodeType, vtype: NodeType, etype: EdgeType) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def z_z_simple_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def z_z_hadamard_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def z_x_simple_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def z_x_hadamard_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def z_boundary_simple_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def z_boundary_hadamard_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def x_x_simple_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def x_x_hadamard_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def x_boundary_simple_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def x_boundary_hadamard_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def boundary_boundary_simple_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def boundary_boundary_hadamard_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        pass

    def num_z_z_simple_edges(self) -> int:
        pass

    def num_z_z_hadamard_edges(self) -> int:
        pass

    def num_z_x_simple_edges(self) -> int:
        pass

    def num_z_x_hadamard_edges(self) -> int:
        pass

    def num_z_boundary_simple_edges(self) -> int:
        pass

    def num_z_boundary_hadamard_edges(self) -> int:
        pass

    def num_x_x_simple_edges(self) -> int:
        pass

    def num_x_x_hadamard_edges(self) -> int:
        pass

    def num_x_boundary_simple_edges(self) -> int:
        pass

    def num_x_boundary_hadamard_edges(self) -> int:
        pass

    def num_boundary_boundary_simple_edges(self) -> int:
        pass

    def num_boundary_boundary_hadamard_edges(self) -> int:
        pass

    def num_edges(self) -> int:
        return self.graph.num_edges()

    def to_undirected_nx_graph(self):
        pass

    def draw(self):
        pass
