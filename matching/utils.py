from collections.abc import Iterator
from typing import Any

from graph.pyzx_nx_conversion import Z_NTYPE_INDEX, X_NTYPE_INDEX, Z_NTYPE_NAME, NTYPE, PHASE, DEGREE
from matching.match import RuleMode


def rule_mode_to_ntype_indices(rule_mode: RuleMode) -> tuple[int, int]:
    return (Z_NTYPE_INDEX, X_NTYPE_INDEX) if rule_mode == Z_NTYPE_NAME else (X_NTYPE_INDEX, Z_NTYPE_INDEX)


def rule_mode_to_ntype_index(rule_mode: RuleMode) -> int:
    return Z_NTYPE_INDEX if rule_mode == Z_NTYPE_NAME else X_NTYPE_INDEX


def node_attributes_equal(v: dict[str, Any], w: dict[str, Any], *args: str) -> bool:
    return all([v[attribute] == w[attribute] for attribute in ([NTYPE, PHASE, DEGREE] if len(args) == 0 else args)])


def filter_permutations(nx_matches: Iterator[dict[int, int]]) -> Iterator[dict[int, int]]:
    matched_pairs = set()
    for match in nx_matches:
        keys = tuple(sorted(list(match.keys())))
        if keys not in matched_pairs:
            matched_pairs.add(keys)
            yield match
