from typing import Union, Tuple, Dict, Any, Generator, TypeVar
from typing_extensions import Literal

from graph.pyzx_nx_conversion import Z_NTYPE_NAME, Z_NTYPE_INDEX, X_NTYPE_INDEX, NTYPE, PHASE, DEGREE

"""
Describes the color mode of a rule. The 'z' mode means that the bottom node of a rule has a z-basis. The remaining
node bases can be deduced from the bottom node basis.
"""
RuleMode = Union[Literal['z'], Literal['x']]


"""class Match:
    def __init__(self, tuple):
        self.tuple = tuple

    def as_dict(self):
        return {self.tuple[i] : i for i in range(len(self.tuple))}"""



FLeftMatch = Tuple[int, int]
FRightMatch = int

YLeftMatch = Tuple[int, int, int, int]
YRightMatch = Tuple[int, int, int, int]

BLeftMatch = Tuple[int, int, int, int]
BRightMatch = Tuple[int, int]

Match = Union[FLeftMatch, FRightMatch, YLeftMatch, YRightMatch, BLeftMatch, BRightMatch]

M = TypeVar('M', Dict[int, int], Match)
Matches = Generator[M, Any, None]


def rule_mode_to_ntype_indices(rule_mode: RuleMode) -> Tuple[int, int]:
    return (Z_NTYPE_INDEX, X_NTYPE_INDEX) if rule_mode == Z_NTYPE_NAME else (X_NTYPE_INDEX, Z_NTYPE_INDEX)


def rule_mode_to_ntype_index(rule_mode: RuleMode) -> int:
    return Z_NTYPE_INDEX if rule_mode == Z_NTYPE_NAME else X_NTYPE_INDEX


def node_attributes_equal(v: Dict[str, Any], w: Dict[str, Any], *args: str) -> bool:
    return all([v[attribute] == w[attribute] for attribute in (args if len(args) == 0 else [NTYPE, DEGREE, PHASE])])


def filter_permutations(matches: Matches[Dict[int, int]]) -> Matches[Dict[int, int]]:
    matched_pairs = set()
    for match in matches:
        print('match = ', match)
        keys = tuple(sorted(list(match)))
        print('keys = ', keys)
        if keys not in matched_pairs:
            matched_pairs.add(keys)
            yield match


def dicts_to_tuples(matches: Matches[Dict[int, int]]) -> Matches[Match]:
    for match in matches:
        yield tuple(match.keys())


def compute_matches() -> Dict[str, Matches]:
    pass
