"""
Tests for Severity Calculation Module
=======================================

Tests calculate_severity, get_severity_level, and build_severity_result
with known masks and boundary conditions.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xai_module.severity_utils import (
    SeverityResult,
    build_severity_result,
    calculate_severity,
    get_severity_level,
)


class TestCalculateSeverity:
    """Tests for mask-based severity calculation."""

    def test_zero_infection(self):
        """All-zero mask should give 0% severity."""
        mask = np.zeros((100, 100), dtype=np.uint8)
        assert calculate_severity(mask) == 0.0

    def test_full_infection(self):
        """All-ones mask should give 100% severity (no leaf mask)."""
        mask = np.ones((100, 100), dtype=np.uint8)
        assert calculate_severity(mask) == 100.0

    def test_partial_infection(self):
        """Known partial mask should give correct percentage."""
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:5, :] = 1  # 50 of 100 pixels
        result = calculate_severity(mask)
        assert abs(result - 50.0) < 0.01

    def test_with_leaf_mask(self):
        """Severity should be relative to leaf area, not total image."""
        infected = np.zeros((10, 10), dtype=np.uint8)
        infected[0:2, 0:5] = 1  # 10 infected pixels

        leaf = np.zeros((10, 10), dtype=np.uint8)
        leaf[0:5, 0:5] = 1  # 25 leaf pixels

        result = calculate_severity(infected, leaf)
        assert abs(result - 40.0) < 0.01  # 10/25 = 40%

    def test_infected_outside_leaf_ignored(self):
        """Infected pixels outside the leaf mask should not count."""
        infected = np.zeros((10, 10), dtype=np.uint8)
        infected[8:10, 8:10] = 1  # 4 infected pixels outside leaf

        leaf = np.zeros((10, 10), dtype=np.uint8)
        leaf[0:5, 0:5] = 1  # 25 leaf pixels, no overlap

        result = calculate_severity(infected, leaf)
        assert result == 0.0

    def test_shape_mismatch_raises(self):
        """Different mask shapes should raise ValueError."""
        infected = np.zeros((10, 10), dtype=np.uint8)
        leaf = np.zeros((20, 20), dtype=np.uint8)

        with pytest.raises(ValueError, match="shape mismatch"):
            calculate_severity(infected, leaf)

    def test_3d_mask_raises(self):
        """3D mask should raise ValueError."""
        mask = np.zeros((10, 10, 3), dtype=np.uint8)

        with pytest.raises(ValueError, match="must be 2D"):
            calculate_severity(mask)

    def test_empty_leaf_mask_raises(self):
        """All-zero leaf mask should raise ValueError (zero area)."""
        infected = np.ones((10, 10), dtype=np.uint8)
        leaf = np.zeros((10, 10), dtype=np.uint8)

        with pytest.raises(ValueError, match="zero"):
            calculate_severity(infected, leaf)


class TestGetSeverityLevel:
    """Tests for severity percentage to level mapping."""

    def test_low(self):
        assert get_severity_level(0.0) == "Low"
        assert get_severity_level(5.0) == "Low"
        assert get_severity_level(9.99) == "Low"

    def test_mild(self):
        assert get_severity_level(10.0) == "Mild"
        assert get_severity_level(20.0) == "Mild"
        assert get_severity_level(29.99) == "Mild"

    def test_moderate(self):
        assert get_severity_level(30.0) == "Moderate"
        assert get_severity_level(45.0) == "Moderate"
        assert get_severity_level(59.99) == "Moderate"

    def test_severe(self):
        assert get_severity_level(60.0) == "Severe"
        assert get_severity_level(80.0) == "Severe"
        assert get_severity_level(100.0) == "Severe"

    def test_out_of_range_raises(self):
        with pytest.raises(ValueError):
            get_severity_level(-1.0)
        with pytest.raises(ValueError):
            get_severity_level(101.0)


class TestBuildSeverityResult:
    """Tests for the unified severity result builder."""

    def test_from_direct_percentage(self):
        result = build_severity_result(severity_percentage=45.0)
        assert result.percentage == 45.0
        assert result.level == "Moderate"
        assert result.source == "direct"

    def test_from_mask(self):
        mask = np.zeros((10, 10), dtype=np.uint8)
        mask[:3, :] = 1  # 30/100 = 30%
        result = build_severity_result(infected_mask=mask)
        assert result.percentage == 30.0
        assert result.level == "Moderate"
        assert result.source == "mask"

    def test_percentage_takes_priority(self):
        """When both percentage and mask are given, percentage wins."""
        mask = np.ones((10, 10), dtype=np.uint8)  # Would be 100%
        result = build_severity_result(severity_percentage=15.0, infected_mask=mask)
        assert result.percentage == 15.0  # Uses direct percentage

    def test_custom_source_label(self):
        result = build_severity_result(severity_percentage=25.0, source="mock")
        assert result.source == "mock"

    def test_neither_raises(self):
        with pytest.raises(ValueError, match="Must provide"):
            build_severity_result()
