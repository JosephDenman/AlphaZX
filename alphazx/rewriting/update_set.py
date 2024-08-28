from typing import NamedTuple

from alphazx.diagram import MatchNode


class UpdateSet(NamedTuple):
    removed_nodes: set[int]
    added_nodes: set[int]
    original_match: MatchNode
