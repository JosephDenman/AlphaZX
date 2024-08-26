import abc
from typing import Type, NamedTuple

import networkx as nx
from typing_extensions import Literal

from alphazx.diagram.constants import S_ETYPE_NAME, I_ETYPE_NAME, SS_ETYPE_NAME, SI_ETYPE_NAME

Basis = Literal['z', 'x']


class ZXMatchDiagramNode(abc.ABC):
    def __hash__(self):
        return hash(self.name)

    @staticmethod
    @property
    @abc.abstractmethod
    def index() -> int:
        pass

    @staticmethod
    @property
    @abc.abstractmethod
    def name() -> str:
        pass

    @staticmethod
    @property
    @abc.abstractmethod
    def abbrev() -> str:
        pass

    @staticmethod
    @property
    @abc.abstractmethod
    def meta_neighbors() -> list[Type['ZXMatchDiagramNode']]:
        pass

    @classmethod
    def is_super_node(cls) -> bool:
        return issubclass(cls, SuperNode)

    @classmethod
    def is_match_node(cls) -> bool:
        return issubclass(cls, MatchNode)


class MatchNode(ZXMatchDiagramNode, abc.ABC):

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

    @staticmethod
    @property
    @abc.abstractmethod
    def expected_size() -> int:
        pass

    @staticmethod
    @property
    @abc.abstractmethod
    def super_node() -> Type['SuperNode']:
        pass

    @classmethod
    def is_simple_match_node(cls) -> bool:
        return issubclass(cls, SimpleMatchNode)

    @classmethod
    def is_basis_match_node(cls) -> bool:
        return issubclass(cls, FRightMatch)

    @classmethod
    def is_compound_match_node(cls) -> bool:
        return issubclass(cls, CompoundMatchNode)

    @property
    def nodes(self) -> list[int]:
        return list(self._nodes)

    @property
    def match(self) -> dict[int, int]:
        return self._match

    @property
    @abc.abstractmethod
    def sub_matches(self) -> list['MatchNode']:
        pass

    def __getitem__(self, item):
        return self._nodes[item]

    def __hash__(self):
        return hash((self.name, *sorted(self.nodes)))

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __repr__(self):
        return self.abbrev + str(list(self._nodes))

    def __iter__(self):
        yield from self._nodes


class SimpleMatchNode(MatchNode, abc.ABC):
    @property
    def node(self) -> int:
        return self.nodes[0]

    @property
    def sub_matches(self) -> list[MatchNode]:
        return []


class CompoundMatchNode(MatchNode, abc.ABC):
    pass


class BoundaryMatch(SimpleMatchNode):
    index = 0
    expected_size = 1
    name = 'boundary'
    abbrev = 'b'

    @staticmethod
    def meta_neighbors() -> list[Type[ZXMatchDiagramNode]]:
        return [BoundarySuperNode, FRightZMatch, FRightXMatch, BoundaryMatch]

    @staticmethod
    def super_node() -> Type[ZXMatchDiagramNode]:
        return BoundarySuperNode


class FRightMatch(SimpleMatchNode, abc.ABC):
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


class FRightZMatch(FRightMatch):
    index = 1
    rule_mode = 'z'
    name = 'f_right_z'
    abbrev = 'frz'

    @staticmethod
    def meta_neighbors() -> list[Type[ZXMatchDiagramNode]]:
        return [FRightZSuperNode, BoundaryMatch, FRightZMatch, FRightXMatch]

    @staticmethod
    def super_node() -> Type[ZXMatchDiagramNode]:
        return FRightZSuperNode


class FRightXMatch(FRightMatch):
    index = 2
    rule_mode = 'x'
    name = 'f_right_x'
    abbrev = 'frx'

    @staticmethod
    def meta_neighbors() -> list[Type[ZXMatchDiagramNode]]:
        return [FRightXSuperNode, BoundaryMatch, FRightZMatch, FRightXMatch]

    @staticmethod
    def super_node() -> Type[ZXMatchDiagramNode]:
        return FRightXSuperNode


def is_boundary(ntype: str | int) -> bool:
    if isinstance(ntype, str):
        return ntype == BoundaryMatch.abbrev
    elif isinstance(ntype, int):
        return ntype == BoundaryMatch.index
    else:
        raise Exception('Unexpected node type representation ' + str(ntype))


def is_z_basis(ntype: str | int) -> bool:
    if isinstance(ntype, str):
        return ntype == FRightZMatch.abbrev
    elif isinstance(ntype, int):
        return ntype == FRightZMatch.index
    else:
        raise Exception('Unexpected node type representation ' + str(ntype))


def is_x_basis(ntype: str | int) -> bool:
    if isinstance(ntype, str):
        return ntype == FRightXMatch.abbrev
    elif isinstance(ntype, int):
        return ntype == FRightXMatch.index
    else:
        raise Exception('Unexpected node type representation ' + str(ntype))


def is_basis(ntype: str | int) -> bool:
    return is_z_basis(ntype) or is_x_basis(ntype)


class FLeftMatch(CompoundMatchNode, abc.ABC):
    expected_size = 2

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


class FLeftZMatch(FLeftMatch):
    index = 3
    rule_mode = 'z'
    name = 'f_left_z'
    abbrev = 'flz'

    @property
    def sub_matches(self) -> list[MatchNode]:
        return [FRightZMatch(node) for node in self.nodes]

    @staticmethod
    def meta_neighbors() -> list[Type[ZXMatchDiagramNode]]:
        return [FLeftZSuperNode, FRightZMatch]

    @staticmethod
    def super_node() -> Type[ZXMatchDiagramNode]:
        return FLeftZSuperNode


class FLeftXMatch(FLeftMatch):
    index = 4
    rule_mode = 'x'
    name = 'f_left_x'
    abbrev = 'flx'

    @property
    def sub_matches(self) -> list[MatchNode]:
        return [FRightXMatch(node) for node in self.nodes]

    @staticmethod
    def meta_neighbors() -> list[Type[ZXMatchDiagramNode]]:
        return [FLeftXSuperNode, FRightXMatch]

    @staticmethod
    def super_node() -> Type[ZXMatchDiagramNode]:
        return FLeftXSuperNode


class BRightMatch(CompoundMatchNode):
    """
    The nodes are ordered as z-x.
    """
    index = 5
    expected_size = 2
    name = 'b_right'
    abbrev = 'br'

    @property
    def sub_matches(self) -> list[MatchNode]:
        return [FRightZMatch(self.nodes[0]), FRightXMatch(self.nodes[1])]

    @staticmethod
    def meta_neighbors() -> list[Type[ZXMatchDiagramNode]]:
        return [BRightSuperNode, FRightZMatch, FRightXMatch]

    @staticmethod
    def super_node() -> Type[ZXMatchDiagramNode]:
        return BRightSuperNode


class BLeftMatch(CompoundMatchNode):
    """
    The nodes are ordered as z-x-z-x.
    """
    index = 6
    expected_size = 4
    name = 'b_left'
    abbrev = 'bl'

    @property
    def sub_matches(self) -> list[MatchNode]:
        z, x, m, n = self.nodes
        return [BRightMatch(z, x), BRightMatch(z, n), BRightMatch(m, x), BRightMatch(m, n), FRightZMatch(z),
                FRightXMatch(x), FRightZMatch(m), FRightXMatch(n)]

    @staticmethod
    def meta_neighbors() -> list[Type[ZXMatchDiagramNode]]:
        return [BLeftSuperNode, BRightMatch, FRightZMatch, FRightXMatch]

    @staticmethod
    def super_node() -> Type[ZXMatchDiagramNode]:
        return BLeftSuperNode


class YRightMatch(CompoundMatchNode, abc.ABC):
    expected_size = 4

    @property
    @abc.abstractmethod
    def rule_mode(self) -> Basis:
        pass


class YRightZMatch(YRightMatch):
    index = 7
    rule_mode = 'z'
    name = 'y_right_z'
    abbrev = 'yrz'

    @property
    def sub_matches(self) -> list[MatchNode]:
        return [FRightXMatch(node) if i == 1 else FRightZMatch(node) for i, node in enumerate(self.nodes)]

    @staticmethod
    def meta_neighbors() -> list[Type[ZXMatchDiagramNode]]:
        return [YRightZSuperNode, FRightZMatch, FRightXMatch]

    @staticmethod
    def super_node() -> Type[ZXMatchDiagramNode]:
        return YRightZSuperNode


class YRightXMatch(YRightMatch):
    index = 8
    rule_mode = 'x'
    name = 'y_right_x'
    abbrev = 'yrx'

    @property
    def sub_matches(self) -> list[MatchNode]:
        return [FRightZMatch(node) if i == 1 else FRightXMatch(node) for i, node in enumerate(self.nodes)]

    @staticmethod
    def meta_neighbors() -> list[Type[ZXMatchDiagramNode]]:
        return [YRightXSuperNode, FRightZMatch, FRightXMatch]

    @staticmethod
    def super_node() -> Type[ZXMatchDiagramNode]:
        return YRightXSuperNode


class YLeftMatch(CompoundMatchNode, abc.ABC):
    expected_size = 4

    @property
    @abc.abstractmethod
    def rule_mode(self) -> Basis:
        pass


class YLeftZMatch(YLeftMatch):
    index = 9
    rule_mode = 'z'
    name = 'y_left_z'
    abbrev = 'ylz'

    @property
    def sub_matches(self) -> list[MatchNode]:
        return [FRightXMatch(node) if i == 1 else FRightZMatch(node) for i, node in enumerate(self.nodes)]

    @staticmethod
    def meta_neighbors() -> list[Type[ZXMatchDiagramNode]]:
        return [YLeftZSuperNode, FRightZMatch, FRightXMatch]

    @staticmethod
    def super_node() -> Type[ZXMatchDiagramNode]:
        return YLeftZSuperNode


class YLeftXMatch(YLeftMatch):
    index = 10
    rule_mode = 'x'
    name = 'y_left_x'
    abbrev = 'ylx'

    @property
    def sub_matches(self) -> list[MatchNode]:
        return [FRightZMatch(node) if i == 1 else FRightXMatch(node) for i, node in enumerate(self.nodes)]

    @staticmethod
    def meta_neighbors() -> list[Type[ZXMatchDiagramNode]]:
        return [YLeftXSuperNode, FRightZMatch, FRightXMatch]

    @staticmethod
    def super_node() -> Type[ZXMatchDiagramNode]:
        return YLeftXSuperNode


class SuperNode(ZXMatchDiagramNode, abc.ABC):

    def __eq__(self, other):
        return self.__hash__() == other.__hash__()

    def __hash__(self):
        return hash(self.index)

    def __repr__(self):
        return self.abbrev

    @staticmethod
    @property
    @abc.abstractmethod
    def sub_node() -> MatchNode:
        pass

    @classmethod
    def meta_neighbors(cls) -> list[Type[ZXMatchDiagramNode]]:
        return list(filter(lambda sub_class: sub_class.name != cls.name, SuperNode.__subclasses__())) + [cls.sub_node]


class BoundarySuperNode(SuperNode):
    index = 11
    sub_node = BoundaryMatch
    name = 'boundary_super'
    abbrev = 'b_super'


class FRightZSuperNode(SuperNode):
    index = 12
    sub_node = FRightZMatch
    name = 'f_right_z_super'
    abbrev = 'frz_super'


class FRightXSuperNode(SuperNode):
    index = 13
    sub_node = FRightXMatch
    name = 'f_right_x_super'
    abbrev = 'frx_super'


class FLeftZSuperNode(SuperNode):
    index = 14
    sub_node = FLeftZMatch
    name = 'f_left_z_super'
    abbrev = 'flz_super'


class FLeftXSuperNode(SuperNode):
    index = 15
    sub_node = FLeftXMatch
    name = 'f_left_x_super'
    abbrev = 'flx_super'


class BRightSuperNode(SuperNode):
    index = 16
    sub_node = BRightMatch
    name = 'b_right_super'
    abbrev = 'br_super'


class BLeftSuperNode(SuperNode):
    index = 17
    sub_node = BLeftMatch
    name = 'b_left_super'
    abbrev = 'bl_super'


class YRightZSuperNode(SuperNode):
    index = 18
    sub_node = YRightZMatch
    name = 'y_right_z_super'
    abbrev = 'yrz_super'


class YRightXSuperNode(SuperNode):
    index = 19
    sub_node = YRightXMatch
    name = 'y_right_x_super'
    abbrev = 'yrx_super'


class YLeftZSuperNode(SuperNode):
    index = 20
    sub_node = YLeftZMatch
    name = 'y_left_z_super'
    abbrev = 'ylz_super'


class YLeftXSuperNode(SuperNode):
    index = 21
    sub_node = YLeftXMatch
    name = 'y_left_x_super'
    abbrev = 'ylx_super'


def from_index_and_node_set(node_type: int, node_set: tuple[int] | list[int] | int) -> MatchNode:
    constructor = METADATA.node_type_constructors[node_type]
    if issubclass(constructor, SimpleMatchNode):
        if isinstance(node_set, tuple) or isinstance(node_set, list):
            assert len(node_set) == 1, f'Expected a single node but received node set {node_set}'
            return constructor(node_set[0])
        else:
            return constructor(node_set)
    elif issubclass(constructor, CompoundMatchNode):
        return constructor(*node_set)
    elif issubclass(constructor, SuperNode):
        return constructor()
    else:
        raise Exception(
            f'Attempted to construct a {constructor} node with node type {node_type} and node set {node_set}')


def _zx_match_diagram_node_leaf_classes() -> list[Type[ZXMatchDiagramNode]]:
    leaf_classes = set()

    def _inner_leaf_classes(cls: Type[ZXMatchDiagramNode]) -> None:
        # If there are no subclasses, this is a leaf
        if not cls.__subclasses__():
            leaf_classes.add(cls)
        else:
            for sub_cls in cls.__subclasses__():
                _inner_leaf_classes(sub_cls)

    _inner_leaf_classes(ZXMatchDiagramNode)
    return sorted(leaf_classes, key=lambda leaf_class: leaf_class.index)


def _count_leaf_classes() -> int:
    return len(_zx_match_diagram_node_leaf_classes())


POSSIBLE_PHASES = [0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1., 1.125, 1.25, 1.375, 1.5, 1.625, 1.75, 1.875]
NUM_POSSIBLE_PHASES = len(POSSIBLE_PHASES)


class Metadata(NamedTuple):
    node_type_abbrevs: list[str]
    node_type_indices: list[int]
    edge_types: list[tuple[str, str, str]]
    node_type_abbrev_index_dict: dict[str, int]
    edge_type_to_index_dict: dict[tuple[str, str, str], int]
    node_type_constructors: list[Type[ZXMatchDiagramNode]]

    basis_node_type_indices: list[int]
    non_basis_node_type_indices: list[int]

    simple_node_type_abbrevs: list[str]
    simple_node_type_indices: list[int]
    simple_edges: list[tuple[str, str, str]]
    non_simple_node_type_indices: list[int]

    super_node_type_abbrevs: list[str]
    super_node_type_indices: list[int]
    non_super_node_type_indices: list[int]

    rewritable_node_types: list[int]

    match_node_type_abbrevs: list[int]
    match_node_type_indices: list[int]
    non_match_node_type_indices: list[int]

    node_feat_to_index_dict: dict[tuple[int, float], int]
    edge_feat_to_index_dict: dict[tuple[int, int], int]

    max_match_size: int
    max_edge_size: int


def compute_meta_edge_type(a: Type[ZXMatchDiagramNode], b: Type[ZXMatchDiagramNode]) -> str:
    if issubclass(a, SuperNode) and issubclass(b, SuperNode):
        return SS_ETYPE_NAME
    elif issubclass(a, SuperNode) and issubclass(b, MatchNode) or issubclass(a, MatchNode) and issubclass(b, SuperNode):
        return SI_ETYPE_NAME
    elif issubclass(a, SimpleMatchNode) and issubclass(b, SimpleMatchNode):
        return S_ETYPE_NAME
    elif issubclass(a, CompoundMatchNode) and issubclass(b, MatchNode) or issubclass(a, MatchNode) and issubclass(b,
                                                                                                                  CompoundMatchNode):
        return I_ETYPE_NAME
    else:
        raise Exception(f'Unexpected relation between node types {a} and {b}')


def _compute_nx_metagraph() -> nx.DiGraph:
    metagraph = nx.DiGraph()
    leaf_classes = _zx_match_diagram_node_leaf_classes()
    for leaf_class in leaf_classes:
        metagraph.add_node(leaf_class, abbrev=leaf_class.abbrev, index=leaf_class.index)
    for leaf_class in leaf_classes:
        for meta_neighbor in leaf_class.meta_neighbors():
            center_e_type = compute_meta_edge_type(leaf_class, meta_neighbor)
            metagraph.add_edge(leaf_class, meta_neighbor,
                               e_type=(leaf_class.abbrev, center_e_type, meta_neighbor.abbrev))
            metagraph.add_edge(meta_neighbor, leaf_class,
                               e_type=(meta_neighbor.abbrev, center_e_type, leaf_class.abbrev))
    return metagraph


def _compute_metadata_from_metagraph(metagraph: nx.DiGraph) -> Metadata:
    node_type_indices = []
    node_type_abbrevs = []
    edge_types = []
    node_type_abbrev_index_dict = {}
    node_type_constructors = []

    simple_node_type_abbrevs = []
    simple_node_type_indices = []
    simple_edges = []
    non_simple_node_type_indices = []

    basis_node_type_indices = []
    non_basis_node_type_indices = []

    super_node_type_abbrevs = []
    super_node_type_indices = []
    non_super_node_type_indices = []
    # super_node_to_simple_node_index_dict = {}

    match_node_type_abbrevs = []
    match_node_type_indices = []
    non_match_node_type_indices = []

    for n, ndata in metagraph.nodes(data=True):
        node_type_abbrev = ndata['abbrev']
        node_type_index = ndata['index']
        node_type_abbrevs.append(node_type_abbrev)
        node_type_indices.append(node_type_index)
        node_type_abbrev_index_dict[node_type_abbrev] = node_type_index
        node_type_constructors.append(n)

    for n, ndata in metagraph.nodes(data=True):
        node_type_abbrev = ndata['abbrev']
        node_type_index = ndata['index']
        if issubclass(n, MatchNode):
            match_node_type_abbrevs.append(node_type_abbrev)
            match_node_type_indices.append(node_type_index)
            # super_node_to_simple_node_index_dict[n.super_node.index] = n.index
        else:
            non_match_node_type_indices.append(node_type_index)

    for n, ndata in metagraph.nodes(data=True):
        node_type_abbrev = ndata['abbrev']
        node_type_index = ndata['index']
        if issubclass(n, SuperNode):
            super_node_type_abbrevs.append(node_type_abbrev)
            super_node_type_indices.append(node_type_index)
        else:
            non_super_node_type_indices.append(node_type_index)

    for n, ndata in metagraph.nodes(data=True):
        node_type_index = ndata['index']
        if issubclass(n, FRightMatch):
            basis_node_type_indices.append(node_type_index)
        else:
            non_basis_node_type_indices.append(node_type_index)

    for n, ndata in metagraph.nodes(data=True):
        node_type_abbrev = ndata['abbrev']
        node_type_index = ndata['index']
        if issubclass(n, SimpleMatchNode):
            simple_node_type_abbrevs.append(node_type_abbrev)
            simple_node_type_indices.append(node_type_index)
        else:
            non_simple_node_type_indices.append(node_type_index)

    for a, b, edata in metagraph.edges(data=True):
        edge_type = edata['e_type']
        edge_types.append(edge_type)
        if edge_type[1] == S_ETYPE_NAME:
            simple_edges.append(edge_type)

    edge_type_to_index_dict = {edge_type: i for i, edge_type in enumerate(edge_types)}

    possible_node_feature_pairs = []
    for n in metagraph.nodes:
        if issubclass(n, FRightMatch):
            for possible_phase in POSSIBLE_PHASES:
                possible_node_feature_pairs.append((n.index, possible_phase))
        else:
            possible_node_feature_pairs.append((n.index, 0.))
    node_feat_to_index_dict = {feature_pair: i for i, feature_pair in enumerate(possible_node_feature_pairs)}

    max_edge_size = 10000
    possible_edge_feature_pairs = []
    for a, b, edata in metagraph.edges(data=True):
        edge_type = edata['e_type']
        if edge_type[1] == S_ETYPE_NAME or edge_types[1] == SI_ETYPE_NAME:
            for possible_edge_size in range(1, max_edge_size + 1):
                possible_edge_feature_pairs.append((edge_type_to_index_dict[edge_type], possible_edge_size))
        else:
            possible_edge_feature_pairs.append((edge_type_to_index_dict[edge_type], 1))
    edge_feat_to_index_dict = {feature_pair: i for i, feature_pair in enumerate(possible_edge_feature_pairs)}

    match_sizes = []
    for n in metagraph.nodes:
        if issubclass(n, MatchNode):
            match_sizes.append(n.expected_size)
    max_match_size = max(match_sizes)

    return Metadata(
        node_type_abbrevs,
        node_type_indices,
        edge_types,
        node_type_abbrev_index_dict,
        edge_type_to_index_dict,
        node_type_constructors,

        basis_node_type_indices,
        non_basis_node_type_indices,

        simple_node_type_abbrevs,
        simple_node_type_indices,
        simple_edges,
        non_simple_node_type_indices,

        super_node_type_abbrevs,
        super_node_type_indices,
        non_super_node_type_indices,
        non_super_node_type_indices + [BoundaryMatch.index],

        match_node_type_abbrevs,
        match_node_type_indices,
        non_match_node_type_indices,

        node_feat_to_index_dict,
        edge_feat_to_index_dict,

        max_match_size,
        max_edge_size)


def _compute_metadata() -> Metadata:
    return _compute_metadata_from_metagraph(_compute_nx_metagraph())


METADATA = _compute_metadata()
