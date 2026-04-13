"""Tests for shared constants module."""

import pytest

from alphazx.shared.constants import ACTION_TYPE_NAMES, N_COMPONENTS_BY_ACTION_TYPE


class TestActionTypeNames:
    def test_has_all_10_types(self):
        assert len(ACTION_TYPE_NAMES) == 10
        assert set(ACTION_TYPE_NAMES.keys()) == set(range(10))

    def test_f_right_types(self):
        assert ACTION_TYPE_NAMES[0] == 'f-right-z'
        assert ACTION_TYPE_NAMES[1] == 'f-right-x'

    def test_all_values_are_strings(self):
        for k, v in ACTION_TYPE_NAMES.items():
            assert isinstance(v, str), f"Key {k} has non-string value {v}"

    def test_all_values_unique(self):
        values = list(ACTION_TYPE_NAMES.values())
        assert len(values) == len(set(values)), "Duplicate action type names"


class TestNComponentsByActionType:
    def test_has_all_10_types(self):
        assert len(N_COMPONENTS_BY_ACTION_TYPE) == 10
        assert set(N_COMPONENTS_BY_ACTION_TYPE.keys()) == set(range(10))

    def test_f_right_has_5_components(self):
        assert N_COMPONENTS_BY_ACTION_TYPE[0] == 5
        assert N_COMPONENTS_BY_ACTION_TYPE[1] == 5

    def test_non_f_right_has_2_components(self):
        for i in range(2, 10):
            assert N_COMPONENTS_BY_ACTION_TYPE[i] == 2, (
                f"Action type {i} ({ACTION_TYPE_NAMES[i]}) should have 2 components"
            )

    def test_all_values_positive_ints(self):
        for k, v in N_COMPONENTS_BY_ACTION_TYPE.items():
            assert isinstance(v, int) and v > 0
