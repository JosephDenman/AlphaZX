import networkx as nx

from diagram.zx_diagram import ZXDiagram
from diagram.zx_match_diagram import ZXMatchDiagram


class PostZXMatchDiagram(ZXMatchDiagram):
    """
    `ZXMatchDiagram` instance that has been processed and is ready to be converted to `Data`.
    """

    def __init__(self, zx_diagram: ZXDiagram, **attr):
        super().__init__(zx_diagram, **attr)
        self.connected_components = sorted(nx.connected_components(self.zx_diagram), key=len, reverse=True)
        component_nodes = []
        for component in self.connected_components:
            self.add_node()
        self.component_nodes = component_nodes
