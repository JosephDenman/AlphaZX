from torch import Tensor
from torch_geometric.data import HeteroData

from diagram.pyzx_graph_generator import nx_clifford_graph
from diagram.zx_diagram import ZXDiagram
from diagram.match import Match
from models.model import MatchType
from rewriting.util import rewrite, FRightParameters

"""
IDEA: Rather than a composition data distribution, allow the probability of an f_right rewrite to be a mixture over all 
      of the f_right nodes, then sample to pick the node. Generate alongside the mixture probabilities the parameters of 
      the distribution from which the additional information is sampled.
"""

"""
How to match the tensor representation of the attributes of a pyg.HeteroData match diagram node with the node from
the ZXMatchDiagram graph? Node identifiers are lost.
"""


def tensor_to_match(action: Tensor) -> tuple[Match, FRightParameters | None]:
    pass


def diagram_value(diagram: ZXDiagram) -> int:
    """
    TODO - Maybe not this simple...
    -1 for every node
    -1 for every edge
    -1 for every non-Clifford gate.
    """
    return - diagram.number_of_nodes() - sum(
        [1 if p % 0.5 != 0 else 0 for p in diagram.phases().values()]) - len(diagram.edges())


def is_simplified(diagram: ZXDiagram) -> bool:
    # TODO - Maybe not this simple...
    return diagram.number_of_nodes() == diagram.num_b_nodes()


def remap_matches(match_dict: dict[MatchType, list[Match]]):
    pass


class ZXGame:
    def __init__(self, num_qubits: int, depth: int, t_gates: bool = True, one_hot_phases: bool = False,
                 one_hot_types: bool = False,
                 simplified_reward: int = 1):
        self.zx_diagram = None
        self.zx_match_diagram = None
        self.previous_value = None
        self.num_qubits = num_qubits
        self.depth = depth
        self.t_gates = t_gates
        self.one_hot_phases = one_hot_phases
        self.one_hot_types = one_hot_types
        self.simplified_reward = simplified_reward

    def step(self, action: Tensor) -> tuple[HeteroData, int, bool]:
        match, params = tensor_to_match(action)
        rewrite(match, self.zx_diagram, params)
        current_value = diagram_value(self.zx_diagram)
        done = is_simplified(self.zx_diagram)  # TODO: What if a graph is repeated? Is there a step limit?
        reward = self.previous_value - current_value + (self.simplified_reward if done else 0)
        self.previous_value = current_value
        # self.zx_match_diagram = ZXMatchDiagram(self.zx_diagram)
        return self.zx_diagram.to_pyg_hetero_data(self.one_hot_types,
                                                  self.one_hot_phases), reward, done

    def reset(self) -> HeteroData:
        self.zx_diagram = ZXDiagram(nx_clifford_graph(self.num_qubits, self.depth, t_gates=self.t_gates))
        # self.zx_match_diagram = ZXMatchDiagram(self.zx_diagram)
        self.previous_value = diagram_value(self.zx_diagram)
        return self.zx_diagram.to_pyg_hetero_data(self.one_hot_types, self.one_hot_phases)
