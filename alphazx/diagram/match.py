import abc
import itertools
from typing import Type

from torch_geometric.typing import NodeType, EdgeType
import torch_geometric as pyg
from typing_extensions import Literal

from alphazx.diagram.constants import S_ETYPE_NAME, I_ETYPE_NAME

Basis = Literal['z', 'x']


class Match(abc.ABC):

    def __init__(self, *match: tuple[int] | list[int] | int):
        """
        :param match: The domain of the dictionary is the nodes from the diagram,
                      while the range of the dictionary is the nodes from the pattern.
        """
        if len(match) == 1:
            self.original_match = {match[0]: 0}
        elif isinstance(match, tuple):
            self.original_match = {node: i for i, node in enumerate(match)}
        elif isinstance(match, list):
            self.original_match = {node: i for i, node in enumerate(tuple(*match))}
        else:
            raise Exception(f'Unexpected argument {match} to match constructor')
        assert len(
            self.original_match) == self.expected_size, \
            f'Constructor for {self.__class__.name} expected {self.expected_size} nodes but received {self.original_match}'
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
    @property
    @abc.abstractmethod
    def abbrev() -> str:
        pass

    @property
    @abc.abstractmethod
    def expected_size(self) -> int:
        pass

    @staticmethod
    @abc.abstractmethod
    def sub_match_types() -> list[Type['Match']]:
        pass

    @classmethod
    def is_simple_match(cls) -> bool:
        return issubclass(cls, SimpleMatch)

    @classmethod
    def is_basis_match(cls) -> bool:
        return issubclass(cls, FRightMatch)

    @property
    def is_compound_match(self) -> bool:
        return not self.is_simple_match

    @property
    def nodes(self) -> list[int]:
        return list(self._nodes)

    @property
    def match(self) -> dict[int, int]:
        return self._match

    @property
    @abc.abstractmethod
    def sub_matches(self) -> list['Match']:
        pass

    def __getitem__(self, item):
        return self._nodes[item]

    def __hash__(self):
        return hash((self.name, *sorted(self.nodes)))

    def __eq__(self, other):
        return isinstance(other, Match) and self.name == other.name and self._nodes == other._nodes

    def __repr__(self):
        return self.abbrev + str(list(self._nodes))

    def __iter__(self):
        yield from self._nodes


class SimpleMatch(Match, abc.ABC):
    @property
    def node(self) -> int:
        return self.nodes[0]

    @property
    def sub_matches(self) -> list[Match]:
        return []


class CompoundMatch(Match, abc.ABC):
    pass


class BoundaryMatch(SimpleMatch):
    name = 'boundary'
    abbrev = 'b'
    index = 0
    expected_size = 1
    sub_match_types = []


def is_boundary(ntype: str | int) -> bool:
    if isinstance(ntype, str):
        return ntype == BoundaryMatch.abbrev
    elif isinstance(ntype, int):
        return ntype == BoundaryMatch.index
    else:
        raise Exception('Unexpected node type representation ' + str(ntype))


class FRightMatch(SimpleMatch, abc.ABC):
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

    @property
    def is_z_basis(self) -> bool:
        return self.rule_mode == 'z'

    @property
    def is_x_basis(self) -> bool:
        return not self.is_z_basis


class FRightZMatch(FRightMatch):
    name = 'f_right_z'
    abbrev = 'z'
    index = 1
    rule_mode = 'z'


def is_z_basis(ntype: str | int) -> bool:
    if isinstance(ntype, str):
        return ntype == FRightZMatch.abbrev
    elif isinstance(ntype, int):
        return ntype == FRightZMatch.index
    else:
        raise Exception('Unexpected node type representation ' + str(ntype))


class FRightXMatch(FRightMatch):
    name = 'f_right_x'
    abbrev = 'x'
    index = 2
    rule_mode = 'x'


def is_x_basis(ntype: str | int) -> bool:
    if isinstance(ntype, str):
        return ntype == FRightXMatch.abbrev
    elif isinstance(ntype, int):
        return ntype == FRightXMatch.index
    else:
        raise Exception('Unexpected node type representation ' + str(ntype))


def is_basis(ntype: str | int) -> bool:
    return is_z_basis(ntype) or is_x_basis(ntype)


class FLeftZMatch(FLeftMatch):
    name = 'f_left_z'
    abbrev = 'zl'
    index = 3
    rule_mode = 'z'
    sub_match_types = [FRightZMatch]

    @property
    def sub_matches(self) -> list[Match]:
        return [FRightZMatch(node) for node in self.nodes]


class FLeftXMatch(FLeftMatch):
    name = 'f_left_x'
    abbrev = 'xl'
    index = 4
    rule_mode = 'x'
    sub_match_types = [FRightXMatch]

    @property
    def sub_matches(self) -> list[Match]:
        return [FRightXMatch(node) for node in self.nodes]


class BRightMatch(CompoundMatch):
    """
    The nodes are ordered as z-x.
    """
    name = 'b_right'
    abbrev = 'br'
    index = 5
    expected_size = 2
    sub_match_types = [FRightZMatch, FRightXMatch]

    @property
    def sub_matches(self) -> list[Match]:
        return [FRightZMatch(self.nodes[0]), FRightXMatch(self.nodes[1])]


class BLeftMatch(CompoundMatch):
    """
    The nodes are ordered as z-x-z-x.
    """
    name = 'b_left'
    abbrev = 'bl'
    index = 6
    expected_size = 4
    sub_match_types = [BRightMatch, FRightZMatch, FRightXMatch]

    @property
    def sub_matches(self) -> list[Match]:
        z, x, m, n = self.nodes
        return [BRightMatch(z, x), BRightMatch(z, n), BRightMatch(m, x), BRightMatch(m, n), FRightZMatch(z),
                FRightXMatch(x), FRightZMatch(m), FRightXMatch(n)]


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
    abbrev = 'yrz'
    index = 7
    rule_mode = 'z'

    @property
    def sub_matches(self) -> list[Match]:
        return [FRightXMatch(node) if i == 1 else FRightZMatch(node) for i, node in enumerate(self.nodes)]


class YLeftZMatch(YLeftMatch):
    name = 'y_left_z'
    abbrev = 'ylz'
    index = 8
    rule_mode = 'z'

    @property
    def sub_matches(self) -> list[Match]:
        return [FRightXMatch(node) if i == 1 else FRightZMatch(node) for i, node in enumerate(self.nodes)]


class YRightXMatch(YRightMatch):
    name = 'y_right_x'
    abbrev = 'yrx'
    index = 9
    rule_mode = 'x'

    @property
    def sub_matches(self) -> list[Match]:
        return [FRightZMatch(node) if i == 1 else FRightXMatch(node) for i, node in enumerate(self.nodes)]


class YLeftXMatch(YLeftMatch):
    name = 'y_left_x'
    abbrev = 'ylx'
    index = 10
    rule_mode = 'x'

    @property
    def sub_matches(self) -> list[Match]:
        return [FRightZMatch(node) if i == 1 else FRightXMatch(node) for i, node in enumerate(self.nodes)]


def from_index_and_node_set(node_type: int, node_set: tuple[int] | list[int] | int) -> Match:
    constructor = INDEX_TO_CONSTRUCTOR_METADATA[node_type]
    if isinstance(constructor, SimpleMatch):
        if isinstance(node_set, tuple) or isinstance(node_set, list):
            assert len(node_set) == 1, f'Expected a single node but received node set {node_set}'
            return constructor(node_set[0])
        else:
            return constructor(node_set)
    else:
        return constructor(*node_set)


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


Metadata = tuple[
    list[NodeType], list[int], list[NodeType], dict[NodeType, int], list[EdgeType], dict[EdgeType, int], list[EdgeType],
    dict[
        EdgeType, int], dict[tuple[int, tuple[float, float]], int], list[Type[Match]], int, list[int]]

POSSIBLE_PHASES = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1., 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875]


def _compute_metadata() -> Metadata:
    edge_metadata = []
    leaf_classes = sorted(_leaf_classes(), key=lambda lc: lc.index)
    node_type_to_index_metadata = {leaf_class.abbrev: leaf_class.index for leaf_class in leaf_classes}
    for leaf_class in leaf_classes:
        sub_match_class_names = [sub_match_class.abbrev for sub_match_class in leaf_class.sub_match_types]
        if len(sub_match_class_names) != 0:
            for sub_match_class_name in sub_match_class_names:
                for a, b in itertools.permutations([leaf_class.abbrev, sub_match_class_name]):
                    edge_metadata.append((a, I_ETYPE_NAME, b))
    simple_leaf_classes = list(filter(lambda lc: lc.is_simple_match(), leaf_classes))
    simple_node_metadata = [leaf_class.abbrev for leaf_class in simple_leaf_classes]
    non_basis_type_indices = [leaf_class.index for leaf_class in
                              list(filter(lambda lc: not lc.is_basis_match(), leaf_classes))]
    simple_edge_metadata = []
    for a, b in itertools.product(simple_node_metadata, simple_node_metadata):
        simple_edge_metadata.append((a, S_ETYPE_NAME, b))
    simple_edge_type_to_index_metadata = {value: index for index, value in enumerate(simple_edge_metadata)}
    edge_metadata += simple_edge_metadata
    edge_type_to_index_metadata = {value: index for index, value in enumerate(edge_metadata)}
    node_metadata = [leaf_class.abbrev for leaf_class in leaf_classes]
    node_type_indices = [leaf_class.index for leaf_class in leaf_classes]
    node_feature_pair_to_index_metadata = {tuple(pair): i for i, pair in
                                           enumerate(list(itertools.product(node_type_indices, POSSIBLE_PHASES)))}
    max_match_size_metadata = max([leaf_class.expected_size for leaf_class in leaf_classes])
    return (node_metadata,
            node_type_indices,
            simple_node_metadata,
            node_type_to_index_metadata,
            edge_metadata,
            edge_type_to_index_metadata,
            simple_edge_metadata,
            simple_edge_type_to_index_metadata,
            node_feature_pair_to_index_metadata,
            leaf_classes,
            max_match_size_metadata,
            non_basis_type_indices)


MATCH_TYPE_COUNT = _count_match_types()

(NODE_METADATA,
 NODE_TYPE_INDICES,
 SIMPLE_NODE_METADATA,
 NODE_TYPE_TO_INDEX_METADATA,
 EDGE_METADATA,
 EDGE_TYPE_TO_INDEX_METADATA,
 SIMPLE_EDGE_METADATA,
 SIMPLE_EDGE_TYPE_TO_INDEX_METADATA,
 NODE_FEATURE_PAIR_TO_INDEX_METADATA,
 INDEX_TO_CONSTRUCTOR_METADATA,
 MAX_MATCH_SIZE_METADATA,
 NON_BASIS_TYPE_INDICES) = _compute_metadata()

SIMPLE_METADATA = SIMPLE_NODE_METADATA, SIMPLE_EDGE_METADATA
METADATA = NODE_METADATA, EDGE_METADATA

print(NODE_TYPE_TO_INDEX_METADATA)