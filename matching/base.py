import abc
from collections import namedtuple
from typing import Union, Tuple, Dict, Any, Generator, TypeVar

import networkx as nx
from typing_extensions import Literal

from graph.pyzx_nx_conversion import Z_NTYPE_NAME, Z_NTYPE_INDEX, X_NTYPE_INDEX, NTYPE, PHASE, DEGREE


"""
Describes the color mode of a rule. The 'z' mode means that the bottom node of a rule has a z-basis. The remaining
node bases can be deduced from the bottom node basis.
"""
RuleMode = Union[Literal['z'], Literal['x']]

FLeftMatch = namedtuple('FLeftMatch', ['left', 'right'])
FRightMatch = namedtuple('FRightMatch', ['center'])

YLeftMatch = namedtuple('YLeftMatch', ['bottom', 'center', 'top_left', 'top_right'])
YRightMatch = namedtuple('YRightMatch', ['bottom', 'center', 'top_left', 'top_right'])

BLeftMatch = namedtuple('BLeftMatch', ['bottom_left', 'bottom_right', 'top_left', 'top_right'])
BRightMatch = namedtuple('BRightMatch', ['bottom', 'top'])

Match = Union[FLeftMatch, FRightMatch, YLeftMatch, YRightMatch, BLeftMatch, BRightMatch]

M = TypeVar('M', Dict[int, int], Match)
Matches = Generator[M, Any, None]


def rule_mode_to_ntype_indices(rule_mode: RuleMode) -> Tuple[int, int]:
    return (Z_NTYPE_INDEX, X_NTYPE_INDEX) if rule_mode == Z_NTYPE_NAME else (X_NTYPE_INDEX, Z_NTYPE_INDEX)


def rule_mode_to_ntype_index(rule_mode: RuleMode) -> int:
    return Z_NTYPE_INDEX if rule_mode == Z_NTYPE_NAME else X_NTYPE_INDEX


def node_attributes_equal(v: Dict[str, Any], w: Dict[str, Any], *args: str) -> bool:
    return all([v[attribute] == w[attribute] for attribute in ([NTYPE, DEGREE, PHASE] if len(args) == 0 else args)])


def sort_dict_by_value(d: Dict[int, int]) -> Dict[int, int]:
    return {k: v for k, v in sorted(d.items(), key=lambda item: item[1])}


def filter_permutations(nx_matches: Matches[Dict[int, int]]) -> Matches[Dict[int, int]]:
    matched_pairs = set()
    for match in nx_matches:
        keys = tuple(sorted(list(match.keys())))
        if keys not in matched_pairs:
            matched_pairs.add(keys)
            yield sort_dict_by_value(match)


def dicts_to_tuples(nx_matches: Matches[Dict[int, int]]) -> Matches[Tuple[..., int]]:
    for match in nx_matches:
        yield tuple(match.keys())


def subgraph_from_match(nx_graph: nx.MultiGraph, match: Match) -> nx.MultiGraph:
    return nx_graph.subgraph(list(match))
