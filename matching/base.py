from typing import Union, Tuple, Dict, Any, Generator
from typing_extensions import Literal

from graph.pyzx_nx_dgl_conversion import Z_NTYPE_NAME, Z_NTYPE_INDEX, X_NTYPE_INDEX

"""
Describes the color mode of a rule. The 'z' mode means that the bottom node of a rule has a z-basis. The remaining
node bases can be deduced from the bottom node basis.
"""
RuleMode = Union[Literal['z'], Literal['x']]

FLeftMatch = Tuple[int, int]
FRightMatch = int

YLeftMatch = Tuple[int, int, int, int]
YRightMatch = Tuple[int, int, int, int]

BLeftMatch = Tuple[int, int, int, int]
BRightMatch = Tuple[int, int]

Match = Union[FLeftMatch, FRightMatch, YLeftMatch, YRightMatch, BLeftMatch, BRightMatch]

Matches = Generator[Match, Any, None]

PHASE = 'phase'
DEGREE = 'degree'
TYPE = 'type'
COLUMN = 'x'
ROW = 'y'


def rule_mode_to_ntype_indices(rule_mode: RuleMode) -> Tuple[int, int]:
    return (Z_NTYPE_INDEX, X_NTYPE_INDEX) if rule_mode == Z_NTYPE_NAME else (X_NTYPE_INDEX, Z_NTYPE_INDEX)


def rule_mode_to_ntype_index(rule_mode: RuleMode) -> int:
    return Z_NTYPE_INDEX if rule_mode == Z_NTYPE_NAME else X_NTYPE_INDEX


def all_node_attributes_equal(v: Dict[str, Any], w: Dict[str, Any]) -> bool:
    return v[TYPE] == w[TYPE] and v[PHASE] == w[PHASE] and v[DEGREE] == w[DEGREE]


def filter_permutations(matches: Matches) -> Matches:
    matched_pairs = set()
    for match in matches:
        keys = tuple(sorted(list(match)))
        if keys not in matched_pairs:
            matched_pairs.add(keys)
            yield match


def compute_matches() -> Dict[str, Matches]:
    pass
