import abc
import itertools
from collections.abc import Iterator
from typing import Type

from torch_geometric.typing import Metadata
from typing_extensions import Literal

from diagram.constants import B_ETYPE_NAME, I_ETYPE_NAME

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

    @staticmethod
    @abc.abstractmethod
    def abbreviated_name() -> str:
        pass

    @property
    @abc.abstractmethod
    def expected_size(self) -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def sub_match_types() -> Iterator[Type['Match']]:
        pass

    @classmethod
    def is_base_match(cls) -> bool:
        return issubclass(cls, BaseMatch)

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


class FRightMatch(BaseMatch, abc.ABC):
    expected_size = 1
    sub_match_types = []

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


class FLeftMatch(CompoundMatch, abc.ABC):
    expected_size = 2
    sub_match_types = []

    @property
    @abc.abstractmethod
    def rule_mode(self) -> Basis:
        pass


class FRightZMatch(FRightMatch):
    name = 'f_right_z'
    abbreviated_name = 'frz'
    index = 0
    rule_mode = 'z'


class FLeftZMatch(FLeftMatch):
    name = 'f_left_z'
    abbreviated_name = 'flz'
    index = 1
    rule_mode = 'z'
    sub_match_types = [FRightZMatch]

    @property
    def sub_matches(self) -> Iterator[Match]:
        for node in self.nodes:
            yield FRightZMatch(node)


class FRightXMatch(FRightMatch):
    name = 'f_right_x'
    abbreviated_name = 'frx'
    index = 2
    rule_mode = 'x'


class FLeftXMatch(FLeftMatch):
    name = 'f_left_x'
    abbreviated_name = 'flx'
    index = 3
    rule_mode = 'x'
    sub_match_types = [FRightXMatch]

    @property
    def sub_matches(self) -> Iterator[Match]:
        for node in self.nodes:
            yield FRightXMatch(node)


class BRightMatch(CompoundMatch):
    name = 'b_right'
    abbreviated_name = 'br'
    index = 4
    expected_size = 2
    sub_match_types = [FRightZMatch, FRightXMatch]

    @property
    def sub_matches(self) -> Iterator[Match]:
        yield FRightXMatch(self.nodes[0])
        yield FRightZMatch(self.nodes[1])


class BLeftMatch(CompoundMatch):
    name = 'b_left'
    abbreviated_name = 'bl'
    index = 5
    expected_size = 4
    sub_match_types = [BRightMatch, FRightZMatch, FRightXMatch]

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


class YRightMatch(CompoundMatch, abc.ABC):
    expected_size = 4
    sub_match_types = [FRightZMatch, FRightXMatch]

    @property
    @abc.abstractmethod
    def rule_mode(self) -> Basis:
        pass


class YLeftMatch(CompoundMatch, abc.ABC):
    expected_size = 4
    sub_match_types = [FRightZMatch, FRightXMatch]

    @property
    @abc.abstractmethod
    def rule_mode(self) -> Basis:
        pass


class YRightZMatch(YRightMatch):
    name = 'y_right_z'
    abbreviated_name = 'yrz'
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
    abbreviated_name = 'ylz'
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
    abbreviated_name = 'yrx'
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
    abbreviated_name = 'ylx'
    index = 9
    rule_mode = 'x'

    @property
    def sub_matches(self) -> Iterator[Match]:
        for node in self._nodes:
            if node == 1:
                yield FRightZMatch(node)
            else:
                yield FRightXMatch(node)


def _leaf_classes() -> set[Type[Match]]:
    leaf_classes = set()

    def _inner_leaf_classes(cls: Type[Match]) -> None:
        # If there are no subclasses, this is a leaf
        if not cls.__subclasses__():
            leaf_classes.add(cls)
        else:
            for sub_cls in cls.__subclasses__():
                _inner_leaf_classes(sub_cls)

    _inner_leaf_classes(Match)
    return leaf_classes


def _count_match_types() -> int:
    return len(_leaf_classes())


def _compute_metadata() -> Metadata:
    node_metadata = []
    edge_metadata = []
    leaf_classes = _leaf_classes()

    for leaf_class in leaf_classes:
        node_metadata.append(leaf_class.abbreviated_name)

    for leaf_class in leaf_classes:
        sub_match_class_names = [sub_match_class.abbreviated_name for sub_match_class in leaf_class.sub_match_types]
        if len(sub_match_class_names) != 0:
            for sub_match_class_name in sub_match_class_names:
                for a, b in itertools.permutations([leaf_class.abbreviated_name, sub_match_class_name]):
                    edge_metadata.append((a, I_ETYPE_NAME, b))

    base_match_names = [leaf_class.abbreviated_name for leaf_class in leaf_classes if leaf_class.is_base_match()]
    for a, b in itertools.permutations(base_match_names, 2):
        edge_metadata.append((a, B_ETYPE_NAME, b))

    return node_metadata, edge_metadata


MATCH_TYPE_COUNT = _count_match_types()
METADATA = _compute_metadata()
