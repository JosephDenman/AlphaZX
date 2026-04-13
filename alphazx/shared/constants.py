"""
Shared constants for AlphaZX training paradigms.
"""

# Human-readable names for action type indices (action_type field in the action tuple).
# action_type 0 → FRightZ (match index 1), action_type 1 → FRightX (match index 2), etc.
ACTION_TYPE_NAMES = {
    0: "f-right-z",
    1: "f-right-x",
    2: "f-left-z",
    3: "f-left-x",
    4: "b-right",
    5: "b-left",
    6: "y-right-z",
    7: "y-left-z",
    8: "y-right-x",
    9: "y-left-x",
}

# Number of active distribution components per action type.
# F-Right uses all 5 heads; all others use only action_type + node.
N_COMPONENTS_BY_ACTION_TYPE = {
    0: 5,  # f-right-z:  action_type + node + phase + new_edge + transfer
    1: 5,  # f-right-x:  action_type + node + phase + new_edge + transfer
    2: 2,  # f-left-z:   action_type + node  (phase/edge/transfer deterministic)
    3: 2,  # f-left-x
    4: 2,  # b-right
    5: 2,  # b-left
    6: 2,  # y-right-z
    7: 2,  # y-left-z
    8: 2,  # y-right-x
    9: 2,  # y-left-x
}
