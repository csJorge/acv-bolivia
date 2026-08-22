"""Tests for sensitivity parameter bounds."""
from __future__ import annotations

import math

import numpy as np
import pytest

from acv_bolivia.core.services.sensitivity_bounds import (
    FALLBACK_BOUNDS_PCT,
    bounds_from_samples,
)


def test_uses_the_observed_range_when_samples_are_valid():
    bounds = bounds_from_samples(
        {"tower": np.array([8.0, 10.0, 12.0])},
        {"tower": 10.0},
    )

    assert bounds == {"tower": (8.0, 12.0)}


def test_uses_nominal_fallback_for_degenerate_samples():
    bounds = bounds_from_samples(
        {"tower": np.array([10.0, 10.0])},
        {"tower": 10.0},
    )

    assert bounds["tower"] == pytest.approx(
        (10.0 * (1 - FALLBACK_BOUNDS_PCT), 10.0 * (1 + FALLBACK_BOUNDS_PCT))
    )


def test_uses_nominal_fallback_for_empty_or_nonfinite_samples():
    bounds = bounds_from_samples(
        {
            "empty": np.array([]),
            "invalid": np.array([math.nan, math.inf]),
        },
        {"empty": 10.0, "invalid": 20.0},
    )

    assert bounds["empty"] == pytest.approx((7.0, 13.0))
    assert bounds["invalid"] == pytest.approx((14.0, 26.0))


def test_includes_nominal_only_components_when_samples_exist():
    bounds = bounds_from_samples(
        {"tower": np.array([8.0, 12.0])},
        {"tower": 10.0, "blades": 20.0},
    )

    assert list(bounds) == ["tower", "blades"]
    assert bounds["blades"] == pytest.approx((14.0, 26.0))


def test_ignores_zero_and_nonfinite_nominals_without_samples():
    bounds = bounds_from_samples(
        None,
        {"zero": 0.0, "nan": math.nan, "infinite": math.inf},
    )

    assert bounds == {}
