import torch

from alphazx.diagram import ZXDiagram
from alphazx.diagram.match import compute_meta_edge_type, ZXMatchDiagramNode, SimpleMatchNode


def compute_edge_type_attr(m: ZXMatchDiagramNode, n: ZXMatchDiagramNode) -> tuple[str, str, str]:
    return m.abbrev, compute_meta_edge_type(m.__class__, n.__class__), n.abbrev


def compute_edge_size_attr(zx_diagram: ZXDiagram, a: ZXMatchDiagramNode, b: ZXMatchDiagramNode) -> torch.Tensor:
    return torch.tensor(
        float(zx_diagram.number_of_edges(a.node, b.node)) if isinstance(a, SimpleMatchNode) and isinstance(b,
                                                                                                           SimpleMatchNode) else 1.)


def compute_node_type_attr(match: ZXMatchDiagramNode) -> str:
    return match.abbrev
