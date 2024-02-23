import abc
from collections.abc import Iterator
from typing import Type

from typing_extensions import Literal

Basis = Literal['z', 'x']


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
        self._match = dict(sorted(self.original_match.items(), key=lambda item: item[1]))
        self._nodes = tuple(self._match.keys())

    @property
    @abc.abstractmethod
    def index(self) -> int:
        pass

    @property
    @abc.abstractmethod
    def name(self) -> str:
        pass

    @property
    @abc.abstractmethod
    def expected_size(self) -> int:
        pass

    @property
    def is_base_match(self) -> bool:
        return isinstance(self, BaseMatch)

    @property
    def is_compound_match(self) -> bool:
        return not self.is_base_match

    @property
    def nodes(self) -> list[int]:
        return list(self._nodes)

    @property
    def match(self) -> dict[int, int]:
        return self._match

    def __getitem__(self, item):
        return self._nodes[item]

    def __hash__(self):
        return hash((self.name, *self._nodes))

    def __eq__(self, other):
        return isinstance(other, Match) and self.name == other.name and self._nodes == other._nodes

    def __repr__(self):
        return self.name + str(list(self._nodes))

    def __iter__(self):
        yield from self._nodes


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
    def rule_mode(self) -> Basis:
        pass

    @property
    def is_z_basis(self) -> bool:
        return self.rule_mode == 'z'

    @property
    def is_x_basis(self) -> bool:
        return not self.is_z_basis


class FLeftMatch(CompoundMatch):
    expected_size = 2

    @property
    @abc.abstractmethod
    def rule_mode(self) -> Basis:
        pass


class YRightMatch(CompoundMatch):
    expected_size = 4

    @property
    @abc.abstractmethod
    def rule_mode(self) -> Basis:
        pass


class YLeftMatch(CompoundMatch):
    expected_size = 4

    @property
    @abc.abstractmethod
    def rule_mode(self) -> Basis:
        pass


class FRightZMatch(FRightMatch):
    name = 'f_right_z'
    index = 0
    rule_mode = 'z'


class FLeftZMatch(FLeftMatch):
    name = 'f_left_z'
    index = 1
    rule_mode = 'z'

    @property
    def sub_matches(self) -> Iterator[Match]:
        for node in self.nodes:
            yield FRightZMatch(node)


class FRightXMatch(FRightMatch):
    name = 'f_right_x'
    index = 2
    rule_mode = 'x'


class FLeftXMatch(FLeftMatch):
    name = 'f_left_x'
    index = 3
    rule_mode = 'x'

    @property
    def sub_matches(self) -> Iterator[Match]:
        for node in self.nodes:
            yield FRightXMatch(node)


class BRightMatch(CompoundMatch):
    name = 'b_right'
    index = 4
    expected_size = 2

    @property
    def sub_matches(self) -> Iterator[Match]:
        yield FRightXMatch(self.nodes[0])
        yield FRightZMatch(self.nodes[1])


class BLeftMatch(CompoundMatch):
    name = 'b_left'
    index = 5
    expected_size = 4

    @property
    def sub_matches(self) -> Iterator[Match]:
        z, x, m, n = self.nodes
        yield BRightMatch(z, x)
        yield BRightMatch(z, n)
        yield BRightMatch(m, x)
        yield BRightMatch(m, n)
        yield FRightZMatch(z)
        yield FRightXMatch(x)
        yield FRightZMatch(m)
        yield FRightXMatch(n)


class YRightZMatch(YRightMatch):
    name = 'y_right_z'
    index = 6
    rule_mode = 'z'

    @property
    def sub_matches(self) -> Iterator[Match]:
        for node in self.nodes:
            if node == 1:
                yield FRightXMatch(node)
            else:
                yield FRightZMatch(node)


class YLeftZMatch(YLeftMatch):
    name = 'y_left_z'
    index = 7
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
    index = 8
    rule_mode = 'x'

    @property
    def sub_matches(self) -> Iterator[Match]:
        for node in self._nodes:
            if node == 1:
                yield FRightZMatch(node)
            else:
                yield FRightXMatch(node)


class YLeftXMatch(YLeftMatch):
    name = 'y_left_x'
    index = 9
    rule_mode = 'x'

    @property
    def sub_matches(self) -> Iterator[Match]:
        for node in self._nodes:
            if node == 1:
                yield FRightZMatch(node)
            else:
                yield FRightXMatch(node)


def count_leaf_subclasses(cls: Type[Match]) -> int:
    """
    Recursively count all leaf subclasses of a given class.

    A leaf subclass is defined as a class that has no further subclasses within its inheritance tree. This function
    traverses the inheritance tree of the specified class, counting only those classes that do not serve as base classes
    for other classes.

    :param cls: The class to count leaf subclasses for. This should be Match or a subclass of Match.
    :return: The total number of leaf subclasses of the given class.
    :rtype: int
    """
    # If there are no subclasses, this is a leaf
    if not cls.__subclasses__():
        return 1
    # Recursively count in subclasses
    return sum(count_leaf_subclasses(sub_cls) for sub_cls in cls.__subclasses__())


MATCH_TYPE_COUNT = count_leaf_subclasses(Match)
