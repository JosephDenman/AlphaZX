from alphazx.diagram.constants import S_ETYPE_NAME, I_ETYPE_NAME
from alphazx.diagram.match import SimpleMatchNode, MatchNode, FRightMatch


def compute_edge_type_attr(m: MatchNode, n: MatchNode) -> tuple[str, str, str]:
    return m.abbrev, S_ETYPE_NAME if isinstance(m, SimpleMatchNode) and isinstance(n, SimpleMatchNode) else I_ETYPE_NAME, n.abbrev


def compute_edge_size_attr(number_of_edges: int, m: MatchNode, n: MatchNode) -> float:
    return float(number_of_edges) if isinstance(m, FRightMatch) and isinstance(n, FRightMatch) else 1.


def compute_node_type_attr(match: MatchNode) -> str:
    return match.abbrev
