"""
Tests for Disease Progression Forecasting Module
===================================================

Tests linear/exponential forecasting, insufficient data handling,
trend classification, and output clamping.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xai_module.disease_forecasting import ForecastResult, forecast_severity


class TestForecastSeverity:
    """Tests for the main forecast function."""

    def test_linear_increasing(self):
        """Linear fit on clearly increasing data should detect upward trend."""
        history = [(1, 10), (3, 20), (5, 30), (7, 40)]
        result = forecast_severity(history, future_days=3, method="linear")

        assert result.reliable is True
        assert result.trend == "Increasing"
        assert result.current_severity == 40.0
        assert len(result.forecast) == 3

        # Predictions should be increasing
        for i in range(1, len(result.forecast)):
            assert result.forecast[i][1] >= result.forecast[i - 1][1]

    def test_linear_decreasing(self):
        """Linear fit on decreasing data should detect downward trend."""
        history = [(1, 60), (3, 45), (5, 30), (7, 15)]
        result = forecast_severity(history, future_days=3, method="linear")

        assert result.reliable is True
        assert result.trend == "Decreasing"

    def test_stable_trend(self):
        """Near-constant data should detect stable trend."""
        history = [(1, 25), (3, 25), (5, 25), (7, 25)]
        result = forecast_severity(history, future_days=3, method="linear")

        assert result.reliable is True
        assert result.trend == "Stable"

    def test_predictions_clamped_to_100(self):
        """Predictions should not exceed 100%."""
        # Steep increase that would extrapolate past 100
        history = [(1, 50), (2, 70), (3, 90)]
        result = forecast_severity(history, future_days=5, method="linear")

        for day, sev in result.forecast:
            assert 0.0 <= sev <= 100.0, f"Day {day}: severity {sev} out of range"

    def test_predictions_clamped_to_0(self):
        """Predictions should not go below 0%."""
        # Steep decrease that would extrapolate below 0
        history = [(1, 30), (2, 15), (3, 5)]
        result = forecast_severity(history, future_days=5, method="linear")

        for day, sev in result.forecast:
            assert 0.0 <= sev <= 100.0, f"Day {day}: severity {sev} out of range"


class TestInsufficientData:
    """Tests for insufficient data handling."""

    def test_empty_history(self):
        """Empty history should return unreliable result."""
        result = forecast_severity([], future_days=3)
        assert result.reliable is False
        assert "Insufficient" in result.message

    def test_single_observation(self):
        """Single data point should return unreliable result."""
        result = forecast_severity([(1, 20)], future_days=3)
        assert result.reliable is False
        assert "Insufficient" in result.message
        assert result.current_severity == 20.0

    def test_two_observations_sufficient(self):
        """Two data points should be enough for a basic forecast."""
        result = forecast_severity([(1, 10), (3, 20)], future_days=3)
        assert result.reliable is True
        assert len(result.forecast) == 3


class TestExponentialForecast:
    """Tests for exponential forecasting method."""

    def test_exponential_method(self):
        """Exponential method should produce valid results."""
        history = [(1, 5), (3, 10), (5, 20), (7, 40)]
        result = forecast_severity(history, future_days=3, method="exponential")

        assert result.reliable is True
        assert len(result.forecast) == 3
        # Exponential growth: later predictions should be larger
        assert result.forecast[-1][1] >= result.forecast[0][1]

    def test_exponential_fallback_with_zeros(self):
        """Exponential should fall back to linear if data contains zeros."""
        history = [(1, 0), (3, 10), (5, 20)]
        result = forecast_severity(history, future_days=3, method="exponential")

        assert result.reliable is True
        # Should have fallen back to linear
        assert "linear" in result.method.lower() or "fallback" in result.method.lower()


class TestForecastResultStructure:
    """Tests for output structure and fields."""

    def test_all_fields_present(self):
        history = [(1, 10), (3, 20), (5, 30)]
        result = forecast_severity(history, future_days=2)

        assert isinstance(result, ForecastResult)
        assert isinstance(result.current_severity, float)
        assert isinstance(result.current_level, str)
        assert isinstance(result.trend, str)
        assert isinstance(result.forecast, list)
        assert isinstance(result.risk_level, str)
        assert isinstance(result.method, str)
        assert isinstance(result.reliable, bool)
        assert isinstance(result.message, str)

    def test_forecast_day_numbers(self):
        """Forecast days should start after the last observation day."""
        history = [(1, 10), (3, 20), (5, 30)]
        result = forecast_severity(history, future_days=3)

        expected_days = [6, 7, 8]
        actual_days = [f[0] for f in result.forecast]
        assert actual_days == expected_days
