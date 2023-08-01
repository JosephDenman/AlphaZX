from collections.abc import Iterator

from matching.b_rule_matcher import b_left_matches, b_right_matches
from matching.f_rule_matcher import f_right_matches, f_left_matches
from matching.match_types import Match
from matching.y_rule_matcher import y_left_matches, y_right_matches
from matching.zx_diagram import ZXDiagram


def compute_matches(diagram: ZXDiagram) -> Iterator[Match]:
    yield from f_right_matches(diagram)
    yield from f_left_matches(diagram)
    yield from b_left_matches(diagram)
    yield from b_right_matches(diagram)
    yield from y_left_matches(diagram)
    yield from y_right_matches(diagram)
