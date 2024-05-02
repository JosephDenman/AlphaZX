from alphazx.diagram.constants import S_ETYPE_NAME, I_ETYPE_NAME
from alphazx.diagram.match import SimpleMatch, Match, FRightMatch


def compute_edge_type_attr(m: Match, n: Match) -> tuple[str, str, str]:
    return m.abbrev, S_ETYPE_NAME if isinstance(m, SimpleMatch) and isinstance(n, SimpleMatch) else I_ETYPE_NAME, n.abbrev


def compute_edge_size_attr(number_of_edges: int, m: Match, n: Match) -> float:
    return float(number_of_edges) if isinstance(m, FRightMatch) and isinstance(n, FRightMatch) else 1.


def compute_node_type_attr(match: Match) -> str:
    return match.abbrev
