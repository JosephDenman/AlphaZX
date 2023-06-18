from collections import defaultdict
from collections.abc import Iterator
from typing import Any

import networkx as nx
import torch
import torch_geometric as pyg

from graph.pyzx_graph_generator import nx_clifford_graph
from graph.pyzx_nx_conv import NTYPE, is_basis, is_z_basis, is_x_basis, ETYPE
from matching.b_rule_matcher import b_left_matches, b_right_matches
from matching.match_types import Match, CompoundMatch, FRightZMatch, FRightXMatch, FRightMatch
from matching.f_rule_matcher import f_left_z_matches, f_right_z_matches, f_left_x_matches, f_right_x_matches
from matching.y_rule_matcher import y_left_z_matches, y_right_z_matches, y_left_x_matches, y_right_x_matches
from matching.zx_diagram import ZXDiagram


class ZXMatchDiagram(nx.Graph):
    NTYPE = 'type'
    NTYPES = 'types'
    PHASES = 'phases'
    DEGREES = 'degrees'

    def __init__(self, diagram: ZXDiagram, **attr):
        self.zx_diagram = diagram
        self.node_attrs = self.zx_diagram.node_attrs
        self.edge_attrs = self.zx_diagram.edge_attrs
        super().__init__(nx.Graph(), **attr)

    @property
    def ntype_idx_to_name(self) -> dict[int, str]:
        return {0: 'f_right_z',
                1: 'f_right_x',
                2: 'f_left_z',
                3: 'f_left_x',
                4: 'b_left',
                5: 'b_right',
                6: 'y_left_z',
                7: 'y_left_x',
                8: 'y_right_z',
                9: 'y_right_x'}

    @property
    def ntype_name_to_idx(self) -> dict[str, int]:
        return {v: k for k, v in self.ntype_idx_to_name}

    @property
    def etype_idx_to_name(self) -> dict[int, (str, str, str)]:
        return {0: ('f_left_z', 'inclusion', 'f_right_z'),
                1: ('f_right_z', 'inclusion', 'f_left_z'),
                2: ('f_left_x', 'inclusion', 'f_right_x'),
                3: ('f_right_x', 'inclusion', 'f_left_x'),
                4: ('b_left', 'inclusion', 'b_right'),
                5: ('b_right', 'inclusion', 'b_left'),
                6: ('b_left', 'inclusion', 'f_right_z'),
                7: ('f_right_z', 'inclusion', 'b_left'),
                8: ('b_left', 'inclusion', 'f_right_x'),
                9: ('f_right_x', 'inclusion', 'b_left'),
                10: ('b_right', 'inclusion', 'f_right_z'),
                11: ('f_right_z', 'inclusion', 'b_right'),
                12: ('b_right', 'inclusion', 'f_right_x'),
                13: ('f_right_x', 'inclusion', 'b_right'),
                14: ('f_right_z', 'inclusion', 'y_left_z'),
                15: ('y_left_z', 'inclusion', 'f_right_z'),
                16: ('f_right_x', 'inclusion', 'y_left_x'),
                17: ('y_left_x', 'inclusion', 'f_right_x'),
                18: ('y_right_z', 'inclusion', 'f_right_z'),
                19: ('f_right_z', 'inclusion', 'y_right_z'),
                20: ('y_right_z', 'inclusion', 'f_right_x'),
                21: ('f_right_x', 'inclusion', 'y_right_z'),
                22: ('y_right_x', 'inclusion', 'f_right_x'),
                23: ('f_right_x', 'inclusion', 'y_right_x'),
                24: ('y_right_x', 'inclusion', 'f_right_z'),
                25: ('f_right_z', 'inclusion', 'y_right_x'),
                26: ('y_left_z', 'inclusion', 'f_right_z'),
                27: ('f_right_z', 'inclusion', 'y_left_z'),
                28: ('y_left_z', 'inclusion', 'f_right_x'),
                29: ('f_right_x', 'inclusion', 'y_left_z'),
                30: ('y_left_x', 'inclusion', 'f_right_x'),
                31: ('f_right_x', 'inclusion', 'y_left_x'),
                32: ('y_left_x', 'inclusion', 'f_right_z'),
                33: ('f_right_z', 'inclusion', 'y_left_x'),
                34: ('f_right_z', 'bridge', 'f_right_x'),
                35: ('f_right_x', 'bridge', 'f_right_z')}

    @property
    def etype_name_to_idx(self) -> dict[(str, str, str), int]:
        return {v: k for k, v in self.etype_idx_to_name}
