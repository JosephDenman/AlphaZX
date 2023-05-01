import dgl
import torch
import networkx as nx
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
        data_dict = {}
        num_nodes_dict = {}
        for t0 in list(NodeType):
            num_nodes_dict[t0.value] = 0
            for t1 in list(NodeType):
                for t2 in list(EdgeType):
                    data_dict[canonical_edge_type(t0, t1, t2)] = ([], [])
        self.graph = dgl.heterograph(data_dict, num_nodes_dict)

    def add_node(self, node_type: NodeType, phase: Optional[float] = None) -> None:
        if phase is not None:
            if node_is_zx(node_type):
                self.graph.add_nodes(1, {'degree': torch.zeros(1), 'phase': torch.tensor([phase])}, node_type.value)
            else:
                raise Exception(f'Attempted to add node type ' + node_type.value + ' with phase ' + str(phase))
        else:
            if not node_is_zx(node_type):
                self.graph.add_nodes(1, ntype=node_type.value)
            else:
                raise Exception('Attempted to add node type ' + node_type.value + ' without a phase')

    def add_z_node(self, phase: float) -> None:
        self.add_node(NodeType.Z, phase)

    def add_x_node(self, phase: float) -> None:
        self.add_node(NodeType.X, phase)

    def add_boundary_node(self) -> None:
        self.add_node(NodeType.BOUNDARY)

    def z_nodes(self) -> int:
        return self.graph.nodes(NodeType.Z.value)

    def x_nodes(self) -> int:
        return self.graph.nodes(NodeType.X.value)

    def boundary_nodes(self) -> int:
        return self.graph.nodes(NodeType.BOUNDARY.value)

    def z_phases(self) -> torch.Tensor:
        return self.graph.nodes[NodeType.Z.value].data['phase']

    def x_phases(self) -> torch.Tensor:
        return self.graph.nodes[NodeType.X.value].data['phase']

    def num_z_nodes(self) -> int:
        return self.graph.number_of_nodes(NodeType.Z.value)

    def num_x_nodes(self) -> int:
        return self.graph.number_of_nodes(NodeType.X.value)

    def num_boundary_nodes(self) -> int:
        return self.graph.number_of_nodes(NodeType.BOUNDARY.value)

    def num_nodes(self) -> int:
        return self.graph.num_nodes()

    def add_edge(self, u: int, utype: NodeType, v: int, vtype: NodeType, etype: EdgeType) -> None:
        uv_cet = canonical_edge_type(utype, vtype, etype)
        vu_cet = canonical_edge_type(vtype, utype, etype)
        if self.graph.has_nodes(u, utype.value):
            if self.graph.has_nodes(v, vtype.value):
                self.graph.add_edges([u], [v], etype=uv_cet)
                self.graph.add_edges([v], [u], etype=vu_cet)
            else:
                raise Exception(f'Node {v} of type {vtype.value} does not exist')
        else:
            raise Exception(f'Node {u} of type {utype.value} does not exist')

    def add_z_z_simple_edge(self, u: int, v: int) -> None:
        self.add_edge(u, NodeType.Z, v, NodeType.Z, EdgeType.SIMPLE)

    def add_z_z_hadamard_edge(self, u: int, v: int) -> None:
        self.add_edge(u, NodeType.Z, v, NodeType.Z, EdgeType.HADAMARD)

    def add_z_x_simple_edge(self, u: int, v: int) -> None:
        self.add_edge(u, NodeType.Z, v, NodeType.X, EdgeType.SIMPLE)

    def add_z_x_hadamard_edge(self, u: int, v: int) -> None:
        self.add_edge(u, NodeType.Z, v, NodeType.X, EdgeType.HADAMARD)

    def add_z_boundary_simple_edge(self, u: int, v: int) -> None:
        self.add_edge(u, NodeType.Z, v, NodeType.BOUNDARY, EdgeType.SIMPLE)

    def add_z_boundary_hadamard_edge(self, u: int, v: int) -> None:
        self.add_edge(u, NodeType.Z, v, NodeType.BOUNDARY, EdgeType.HADAMARD)

    def add_x_x_simple_edge(self, u: int, v: int) -> None:
        self.add_edge(u, NodeType.X, v, NodeType.X, EdgeType.SIMPLE)

    def add_x_x_hadamard_edge(self, u: int, v: int) -> None:
        self.add_edge(u, NodeType.X, v, NodeType.X, EdgeType.HADAMARD)

    def add_x_boundary_simple_edge(self, u: int, v: int) -> None:
        self.add_edge(u, NodeType.X, v, NodeType.BOUNDARY, EdgeType.SIMPLE)

    def add_x_boundary_hadamard_edge(self, u: int, v: int) -> None:
        self.add_edge(u, NodeType.X, v, NodeType.BOUNDARY, EdgeType.HADAMARD)

    def add_boundary_boundary_simple_edge(self, u: int, v: int) -> None:
        self.add_edge(u, NodeType.BOUNDARY, v, NodeType.BOUNDARY, EdgeType.SIMPLE)

    def add_boundary_boundary_hadamard_edge(self, u: int, v: int) -> None:
        self.add_edge(u, NodeType.BOUNDARY, v, NodeType.BOUNDARY, EdgeType.HADAMARD)

    def edges(self, utype: NodeType, vtype: NodeType, etype: EdgeType) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.graph.edges(etype=canonical_edge_type(utype, vtype, etype))

    def z_z_simple_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.edges(NodeType.Z, NodeType.Z, EdgeType.SIMPLE)

    def z_z_hadamard_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.edges(NodeType.Z, NodeType.Z, EdgeType.HADAMARD)

    def z_x_simple_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.edges(NodeType.Z, NodeType.X, EdgeType.SIMPLE)

    def z_x_hadamard_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.edges(NodeType.Z, NodeType.X, EdgeType.HADAMARD)

    def z_boundary_simple_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.edges(NodeType.Z, NodeType.BOUNDARY, EdgeType.SIMPLE)

    def z_boundary_hadamard_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.edges(NodeType.Z, NodeType.BOUNDARY, EdgeType.HADAMARD)

    def x_x_simple_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.edges(NodeType.X, NodeType.X, EdgeType.SIMPLE)

    def x_x_hadamard_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.edges(NodeType.X, NodeType.X, EdgeType.HADAMARD)

    def x_boundary_simple_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.edges(NodeType.X, NodeType.BOUNDARY, EdgeType.SIMPLE)

    def x_boundary_hadamard_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.edges(NodeType.X, NodeType.BOUNDARY, EdgeType.HADAMARD)

    def boundary_boundary_simple_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.edges(NodeType.BOUNDARY, NodeType.BOUNDARY, EdgeType.SIMPLE)

    def boundary_boundary_hadamard_edges(self) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.edges(NodeType.BOUNDARY, NodeType.BOUNDARY, EdgeType.HADAMARD)

    def num_z_z_simple_edges(self) -> int:
        return self.graph.num_edges(canonical_edge_type(NodeType.Z, NodeType.Z, EdgeType.SIMPLE))

    def num_z_z_hadamard_edges(self) -> int:
        return self.graph.num_edges(canonical_edge_type(NodeType.Z, NodeType.Z, EdgeType.HADAMARD))

    def num_z_x_simple_edges(self) -> int:
        return self.graph.num_edges(canonical_edge_type(NodeType.Z, NodeType.X, EdgeType.SIMPLE))

    def num_z_x_hadamard_edges(self) -> int:
        return self.graph.num_edges(canonical_edge_type(NodeType.Z, NodeType.X, EdgeType.HADAMARD))

    def num_z_boundary_simple_edges(self) -> int:
        return self.graph.num_edges(canonical_edge_type(NodeType.Z, NodeType.BOUNDARY, EdgeType.SIMPLE))

    def num_z_boundary_hadamard_edges(self) -> int:
        return self.graph.num_edges(canonical_edge_type(NodeType.Z, NodeType.BOUNDARY, EdgeType.HADAMARD))

    def num_x_x_simple_edges(self) -> int:
        return self.graph.num_edges(canonical_edge_type(NodeType.X, NodeType.X, EdgeType.SIMPLE))

    def num_x_x_hadamard_edges(self) -> int:
        return self.graph.num_edges(canonical_edge_type(NodeType.X, NodeType.X, EdgeType.HADAMARD))

    def num_x_boundary_simple_edges(self) -> int:
        return self.graph.num_edges(canonical_edge_type(NodeType.X, NodeType.BOUNDARY, EdgeType.SIMPLE))

    def num_x_boundary_hadamard_edges(self) -> int:
        return self.graph.num_edges(canonical_edge_type(NodeType.X, NodeType.BOUNDARY, EdgeType.HADAMARD))

    def num_boundary_boundary_simple_edges(self) -> int:
        return self.graph.num_edges(canonical_edge_type(NodeType.BOUNDARY, NodeType.BOUNDARY, EdgeType.SIMPLE))

    def num_boundary_boundary_hadamard_edges(self) -> int:
        return self.graph.num_edges(canonical_edge_type(NodeType.BOUNDARY, NodeType.BOUNDARY, EdgeType.HADAMARD))

    def num_edges(self) -> int:
        return self.graph.num_edges()

    def to_undirected_nx_graph(self):
        return dgl.to_networkx(dgl.to_homogeneous(self.graph)).to_undirected()

    def draw(self):
        import matplotlib.pyplot as plt
        z_phases = self.z_phases()
        x_phases = self.x_phases()
        # z nodes
        # boundary nodes
        # edges with types
        homograph = dgl.to_homogeneous(self.graph)
        nx_graph = dgl.to_networkx(homograph)
        undirected_nx_graph = nx_graph.to_undirected()
        print(undirected_nx_graph.nodes(data=True))

        nx.draw(undirected_nx_graph, with_labels=True, pos=nx.spring_layout(undirected_nx_graph))
        plt.show()
