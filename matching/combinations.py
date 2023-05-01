from typing import Union, Literal

BRuleNodeTypes = Union[Literal['b_left_z'], Literal['b_left_x'], Literal['b_right_z'], Literal['b_right_x']]

YRuleNodeTypes = Union[Literal['y_left_z'], Literal['y_left_x'], Literal['y_right_z'], Literal['y_right_x']]

FRuleNodeTypes = Union[Literal['f_left_z'], Literal['f_left_x'], Literal['f_right_z'], Literal['f_right_x']]

SubgraphTypes = Union[BRuleNodeTypes, YRuleNodeTypes, FRuleNodeTypes]

MetagraphTypes = Union[BRuleNodeTypes, YRuleNodeTypes, FRuleNodeTypes]
