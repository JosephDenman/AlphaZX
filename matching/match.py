import abc
from collections.abc import Iterator
from typing import Any, TypeVar, Generic

import networkx as nx
from typing_extensions import Literal

"""
Describes the color mode of a rule. The 'z' mode means that the bottom node of a rule has a z-basis. The remaining
node bases can be deduced from the bottom node basis.
"""
RuleMode = Literal['z', 'x']


def camel_to_snake(s: str) -> str:
    res = ""
    for i in s:
        if i.isupper():
            res += "_" + i.lower()
        else:
            res += i
    return res[1:]


class Match(abc.ABC):

    def __init__(self, *match: dict[int, int] | int):
        """
        :param match: The domain of the dictionary is the nodes from the diagram,
                      while the range of the dictionary is the nodes from the pattern.
        """
        if len(match) == 1 and isinstance(match[0], dict):
            self.original_match = match[0]
        elif isinstance(match, tuple):
            self.original_match = {node: i for i, node in enumerate(match)}
        else:
            raise Exception(f'Unexpected argument {match} to match constructor')
        assert len(
            self.original_match) == self.expected_size, \
            f'Expected {self.expected_size} nodes but received {len(self.original_match)}'
        self.match = dict(sorted(self.original_match.items(), key=lambda item: item[1]))
        self.nodes = tuple(self.match.keys())

    @property
    @abc.abstractmethod
    def index(self):
        pass

    @property
    @abc.abstractmethod
    def name(self):
        pass

    @property
    @abc.abstractmethod
    def expected_size(self):
        pass

    @property
    def is_base_match(self) -> bool:
        return isinstance(self, BaseMatch)

    @property
    def is_compound_match(self) -> bool:
        return not self.is_base_match

    def __getitem__(self, item):
        return self.nodes[item]

    def __hash__(self):
        return hash((self.name, *self.nodes))

    def __eq__(self, other):
        return isinstance(other, Match) and self.name == other.name and self.nodes == other.nodes

    def __repr__(self):
        return self.name + str(list(self.nodes))

    def __iter__(self):
        yield from self.nodes


class BaseMatch(Match, abc.ABC):
    pass


class CompoundMatch(Match, abc.ABC):
    @property
    @abc.abstractmethod
    def sub_matches(self) -> Iterator[Match]:
        pass


class FRightMatch(BaseMatch):
    expected_size = 1

    @property
    @abc.abstractmethod
    def rule_mode(self) -> RuleMode:
        pass


class FRightZMatch(FRightMatch):
    name = 'f_right_z'
    index = 0
    rule_mode = 'z'


class FRightXMatch(FRightMatch):
    name = 'f_right_x'
    index = 1
    rule_mode = 'x'


class FLeftMatch(CompoundMatch):
    expected_size = 2

    @property
    @abc.abstractmethod
    def rule_mode(self) -> RuleMode:
        pass


class FLeftZMatch(FLeftMatch):
    name = 'f_left_z'
    index = 2
    rule_mode = 'z'

    @property
    def sub_matches(self) -> Iterator[Match]:
        for node in self.nodes:
            yield FRightZMatch(node)


class FLeftXMatch(FLeftMatch):
    name = 'f_left_x'
    index = 3
    rule_mode = 'x'

    @property
    def sub_matches(self) -> Iterator[Match]:
        for node in self.nodes:
            yield FRightXMatch(node)


class BLeftMatch(CompoundMatch):
    name = 'b_left'
    index = 4
    expected_size = 4

    @property
    def sub_matches(self) -> Iterator[Match]:
        for z in self.nodes[:2]:
            for x in self.nodes[2:]:
                yield BRightMatch(z, x)
        yield FRightZMatch(self.nodes[0])
        yield FRightZMatch(self.nodes[1])
        yield FRightXMatch(self.nodes[2])
        yield FRightXMatch(self.nodes[3])


class BRightMatch(CompoundMatch):
    name = 'b_right'
    index = 5
    expected_size = 2

    @property
    def sub_matches(self) -> Iterator[Match]:
        yield FRightZMatch(self.nodes[0])
        yield FRightXMatch(self.nodes[1])


class YLeftMatch(CompoundMatch):
    expected_size = 4

    @property
    @abc.abstractmethod
    def rule_mode(self) -> RuleMode:
        pass


class YLeftZMatch(YLeftMatch):
    name = 'y_left_z'
    index = 6
    rule_mode = 'z'

    @property
    def sub_matches(self) -> Iterator[Match]:
        for node in self.nodes:
            if node == 1:
                yield FRightXMatch(node)
            else:
                yield FRightZMatch(node)


class YLeftXMatch(YLeftMatch):
    name = 'y_left_x'
    index = 7
    rule_mode = 'x'

    @property
    def sub_matches(self) -> Iterator[Match]:
        for node in self.nodes:
            if node == 1:
                yield FRightZMatch(node)
            else:
                yield FRightXMatch(node)


class YRightMatch(CompoundMatch):
    expected_size = 4

    @property
    @abc.abstractmethod
    def rule_mode(self) -> RuleMode:
        pass


class YRightZMatch(YRightMatch):
    name = 'y_right_z'
    index = 8
    rule_mode = 'z'

    @property
    def sub_matches(self) -> Iterator[Match]:
        for node in self.nodes:
            if node == 1:
                yield FRightXMatch(node)
            else:
                yield FRightZMatch(node)


class YRightXMatch(YRightMatch):
    name = 'y_right_x'
    index = 9
    rule_mode = 'x'

    @property
    def sub_matches(self) -> Iterator[Match]:
        for node in self.nodes:
            if node == 1:
                yield FRightZMatch(node)
            else:
                yield FRightXMatch(node)


M = TypeVar('M', bound=Match)


class Matcher(abc.ABC, Generic[M]):

    @property
    @abc.abstractmethod
    def pattern(self) -> nx.MultiGraph:
        pass

    @abc.abstractmethod
    def matches(self, diagram: nx.MultiGraph) -> Iterator[M]:
        pass


class Rule(abc.ABC, Generic[M]):

    @property
    @abc.abstractmethod
    def matcher(self) -> Matcher[M]:
        pass

    @abc.abstractmethod
    def rewrite(self, diagram: nx.MultiGraph, match: M) -> None:
        pass


class RuleSet:
    # Test that all constituent rules have different names and indices
    pass
