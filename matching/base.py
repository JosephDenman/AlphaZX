import abc
from abc import ABC
from collections import namedtuple
from collections.abc import Iterator
from typing import Any, TypeVar, Generic, Union

import networkx as nx
from typing_extensions import Literal

from graph.pyzx_nx_conversion import Z_NTYPE_NAME, Z_NTYPE_INDEX, X_NTYPE_INDEX, NTYPE, PHASE, DEGREE


"""
Describes the color mode of a rule. The 'z' mode means that the bottom node of a rule has a z-basis. The remaining
node bases can be deduced from the bottom node basis.
"""
RuleMode = Literal['z', 'x']


class Match(abc.ABC):

    def __init__(self, match: dict[int, int]):
        self.match = dict(sorted(match.items(), key=lambda item: item[1]))
        self.nodes = tuple(self.match.keys())

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    def __hash__(self):
        return hash((self.name, *self.nodes))

    def __eq__(self, other):
        return isinstance(other, Match) and self.name == other.name and self.nodes == other.nodes


class BaseMatch(Match, abc.ABC):

    @property
    def is_base_match(self) -> bool:
        return True

    def __eq__(self, other):
        return isinstance(other, Match) and self.name == other.name and self.nodes == other.nodes


class CompoundMatch(Match, abc.ABC):

    @property
    def is_base_match(self) -> bool:
        return False

    @property
    @abc.abstractmethod
    def sub_matches(self) -> Iterator[Match]:
        pass


class FLeftMatchZ(BaseMatch):

    name = 'f_left_match_z'

    def __getitem__(self, item):
        return self.match[item]


class FLeftMatchX(BaseMatch):

    name = 'f_left_match_x'

    def __getitem__(self, item):
        return self.match[item]

    def test(self) -> None:
        print('test = ', isinstance(self, BaseMatch))



M = TypeVar('M', bound=Match)


class Rule(abc.ABC, Generic[M]):

    @abc.abstractmethod
    def matcher(self, diagram: nx.MultiGraph) -> Iterator[M]:
        pass

    @abc.abstractmethod
    def rewrite(self, diagram: nx.MultiGraph, match: M) -> None:
        pass


class RuleSet:
    "Test that all constituent rules have different names"
    pass


FLeftMatch = namedtuple('FLeftMatch', ['a', 'b'])
FRightMatch = namedtuple('FRightMatch', ['a'])

YLeftMatch = namedtuple('YLeftMatch', ['a', 'b', 'c', 'd'])
YRightMatch = namedtuple('YRightMatch', ['a', 'b', 'c', 'd'])

BLeftMatch = namedtuple('BLeftMatch', ['a', 'b', 'c', 'd'])
BRightMatch = namedtuple('BRightMatch', ['a', 'b'])

Match = FLeftMatch | FRightMatch | YLeftMatch | YRightMatch | BLeftMatch | BRightMatch


def rule_mode_to_ntype_indices(rule_mode: RuleMode) -> tuple[int, int]:
    return (Z_NTYPE_INDEX, X_NTYPE_INDEX) if rule_mode == Z_NTYPE_NAME else (X_NTYPE_INDEX, Z_NTYPE_INDEX)


def rule_mode_to_ntype_index(rule_mode: RuleMode) -> int:
    return Z_NTYPE_INDEX if rule_mode == Z_NTYPE_NAME else X_NTYPE_INDEX


def node_attributes_equal(v: dict[str, Any], w: dict[str, Any], *args: str) -> bool:
    return all([v[attribute] == w[attribute] for attribute in ([NTYPE, PHASE] if len(args) == 0 else args)])


def filter_permutations(nx_matches: Iterator[dict[int, int]]) -> Iterator[dict[int, int]]:
    matched_pairs = set()
    for match in nx_matches:
        keys = tuple(sorted(list(match.keys())))
        if keys not in matched_pairs:
            matched_pairs.add(keys)
            yield match


def subgraph_from_match(nx_graph: nx.MultiGraph, match: Match) -> nx.MultiGraph:
    return nx_graph.subgraph(list(match))
