
"""

There are three obvious possible representations for the state graph. Each has implications for the input to the 
prediction model.

------------------------------------------------------------------------------------------------------------------------
# Homogenous Direct Diagram Encoding

The first approach is to directly convert the `ZXDiagram` to a `Data` object. This requires that node feature dimensions 
are the same for all node types. The representations are the concatenation of an integer representing the type of the
vertex, the phase of the vertex (zero for input and output vertices), and the degree of the vertex.

    i_feat = [0] + [0] + [1]

    o_feat = [1] + [0] + [1]
    
    z_feat = [2] + [phase] + [degree]
    
    x_feat = [3] + [phase] + [degree]

The vertex types are passed to an embedding layer. This representation allows using GPS, which allows positional/structural 
encoding techniques that are generally not available to HGNNs.

The main obstacle the direct representation faces is that of action encodings. In AlphaZX, an action is a rewrite match
identifying subgraph of G. The node outputs of the GNN must be transformed into a probability distribution over matches. 
In contrast to the match diagram encodings, the direct encoding requires a mechanism to assemble match-level outputs 
from node-level outputs in an end-to-end fashion. The mechanism must also support batching. One possibility is to provide 
the match sets as inputs to the model:

    def forward(x, 
                edge_index, 
                edge_attr, 
                flz_matches, 
                flx_matches, 
                frz_matches, frx_matches, bl_matches, br_matches, ylz_matches, yrz_matches, ylx_matches, yrx_matches)
        return value, policy_params
                        
Each match `m` is a mask that selects the elements of `x` contained in `m`. The output features can then be passed to
a pooling layer:

    `h_S_i = Pool_t(S)(h[m])`

# Policy


    
------------------------------------------------------------------------------------------------------------------------
# Homogenous Match Diagram Encoding

    flz_feat = []
    
    flx_feat = []
    
    frz_feat = []
    
    frx_feat = []
    
    bl_feat = []
    
    br_feat = []
    
    ylz_feat = []
    
    yrz_feat = []
    
    ylx_feat = []
    
    yrx_feat = []

------------------------------------------------------------------------------------------------------------------------

Observations:

    O1: Z or X gates that (1) have zero phase, (2) are the only gates on a circuit layer, and (3) belong to a circuit layer
       that does not interact with other circuit layers can be removed from the graph. This is not true when the objective is to minimize the number
        of gates with a particular phase set.

    O2: Z or X gates that satisfy only (O1.2) and (O1.3) above can be excluded from the model, since expanding these spiders 
        always produce a circuit that is sub-optimally larger. This is not true when the objective is to minimize the number
        of gates with a particular phase set.

    O3: Single isolated vertices in a match diagram always describe vertices satisfying at least (O1.2) and (O1.3).
       
Challenges:

    P1: ZX diagrams typically have multiple connected components due to the way they are constructed. Circuits are essentially
        a set of stacked, disconnected, horizontal lines, each of which begins with an input node and ends with an output 
        node. To these horizontal lines, a small number of vertical lines are added at some nodes in the middle. As a result
        of diagrams not being completely connected, messages will not propagate to all nodes. Messages are only propagated
        within connected components.
       
    S1: For each connected component C_i in G, add a component-level vertex c_i of type CO and connect it to each vertex 
        in C_i with an edge of type e_CO-VE. Connect each component-level vertex c_i to each other component-level vertex
        c_j with an edge of type e_CO-CO. By connecting component-level vertices to each other rather than a single top-level
        vertex, messages coming from different components can be distinguished.
       
    S2: Given node representations produced from running GNNs on each component (no additional logic is required to restrict
        the GNN to each component, since this is handled by disconnectedness) apply a transformer encoder to all final node
        representations. Transformer encoders work pair-wise globally, meaning that features are not propagated through
        component-level vertex bottlenecks.

------------------------------------------------------------------------------------------------------------------------

Decisions:

    D1: Use https://arxiv.org/pdf/2108.13650.pdf as a HGNN-specific feature encoder with hetero-transformed GPSModule.

"""