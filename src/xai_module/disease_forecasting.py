"""
Disease Progression Forecasting
=================================

Estimates future disease severity based on historical severity observations.
Supports linear and exponential trend extrapolation.

When historical data is insufficient (< 2 observations), the system returns
an explicit warning rather than producing unreliable predictions.

Author: Vidhi Garg
"""

import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .config import (
    FORECAST_DEFAULT_FUTURE_DAYS,
    FORECAST_DEFAULT_METHOD,
    FORECAST_MIN_DATA_POINTS,
    SEVERITY_THRESHOLDS,
)
from .severity_utils import get_severity_level


@dataclass
class ForecastResult:
    """
    Structured output from disease progression forecasting.

    Attributes:
        current_severity: The most recent observed severity percentage.
        current_level: Severity category for the current observation.
        trend: Direction of disease progression ("Increasing", "Stable", "Decreasing").
        forecast: List of (day, predicted_severity_pct) tuples for future days.
        risk_level: Combined risk assessment based on severity + trend.
        method: Forecasting method used ("linear", "exponential").
        reliable: Whether the forecast has enough data to be considered reliable.
        message: Human-readable summary or warning message.
    """
    current_severity: float
    current_level: str
    trend: str
    forecast: List[Tuple[int, float]]
    risk_level: str
    method: str
    reliable: bool
    message: str


def _classify_trend(slope: float, threshold: float = 0.5) -> str:
    """
    Classify the trend direction based on the fitted slope.

    Args:
        slope: The slope of the fitted line (severity % per day).
        threshold: Minimum absolute slope to classify as non-stable.

    Returns:
        "Increasing", "Stable", or "Decreasing".
    """
    if slope > threshold:
        return "Increasing"
    elif slope < -threshold:
        return "Decreasing"
    else:
        return "Stable"


def _get_risk_level(severity_level: str, trend: str) -> str:
    """
    Derive a risk level from the current severity category and trend direction.

    Uses the RISK_LEVEL_MAP from config, with a sensible fallback.
    """
    from .config import RISK_LEVEL_MAP

    return RISK_LEVEL_MAP.get((severity_level, trend), "Unknown")


def _fit_linear(days: np.ndarray, severities: np.ndarray):
    """Fit a linear model: severity = slope * day + intercept."""
    coeffs = np.polyfit(days, severities, deg=1)
    slope, intercept = coeffs[0], coeffs[1]
    return slope, intercept


def _fit_exponential(days: np.ndarray, severities: np.ndarray):
    """
    Fit an exponential model: severity = a * exp(b * day).

    Uses log-transform + linear fit. Falls back to linear if any
    severity values are <= 0 (log undefined).

    Returns:
        (a, b) parameters, or None if exponential fit is not feasible.
    """
    if np.any(severities <= 0):
        return None

    log_sev = np.log(severities)
    try:
        coeffs = np.polyfit(days, log_sev, deg=1)
        b = coeffs[0]
        a = np.exp(coeffs[1])
        return a, b
    except (np.RankWarning, ValueError):
        return None


def forecast_severity(
    history: List[Tuple[int, float]],
    future_days: Optional[int] = None,
    method: Optional[str] = None,
) -> ForecastResult:
    """
    Forecast future disease severity based on historical observations.

    Args:
        history: List of (day_number, severity_percentage) tuples.
                 Example: [(1, 10), (3, 18), (5, 31), (7, 45)]
        future_days: Number of days to forecast ahead from the last observation.
                     Defaults to FORECAST_DEFAULT_FUTURE_DAYS from config.
        method: Forecasting method — "linear" or "exponential".
                Defaults to FORECAST_DEFAULT_METHOD from config.
                If "exponential" fails (e.g., zero severities), falls back to "linear".

    Returns:
        ForecastResult with predictions, trend, risk level, and reliability flag.
    """
    if future_days is None:
        future_days = FORECAST_DEFAULT_FUTURE_DAYS
    if method is None:
        method = FORECAST_DEFAULT_METHOD

    # ---- Insufficient data check ----
    if len(history) < FORECAST_MIN_DATA_POINTS:
        # Return a result with explicit warning
        current_sev = history[-1][1] if history else 0.0
        current_lvl = get_severity_level(min(max(current_sev, 0.0), 100.0))
        return ForecastResult(
            current_severity=current_sev,
            current_level=current_lvl,
            trend="Unknown",
            forecast=[],
            risk_level="Unknown",
            method=method,
            reliable=False,
            message=(
                "Insufficient historical observations for reliable progression "
                f"forecasting. Need at least {FORECAST_MIN_DATA_POINTS} data points, "
                f"got {len(history)}."
            ),
        )

    # Convert to numpy arrays
    days = np.array([h[0] for h in history], dtype=float)
    severities = np.array([h[1] for h in history], dtype=float)

    # Current (most recent) observation
    last_day = int(days[-1])
    current_sev = float(severities[-1])
    current_lvl = get_severity_level(min(max(current_sev, 0.0), 100.0))

    # ---- Fit model ----
    used_method = method
    slope = 0.0

    if method == "exponential":
        exp_params = _fit_exponential(days, severities)
        if exp_params is not None:
            a, b = exp_params
            # Generate predictions
            future_day_nums = list(range(last_day + 1, last_day + future_days + 1))
            predictions = []
            for d in future_day_nums:
                pred = a * np.exp(b * d)
                pred = float(np.clip(pred, 0.0, 100.0))
                predictions.append((d, round(pred, 2)))
            # Estimate trend from effective linear slope at the current point
            slope = a * b * np.exp(b * last_day)
        else:
            # Fallback to linear
            used_method = "linear (fallback from exponential)"
            method = "linear"

    if method == "linear":
        slope, intercept = _fit_linear(days, severities)
        future_day_nums = list(range(last_day + 1, last_day + future_days + 1))
        predictions = []
        for d in future_day_nums:
            pred = slope * d + intercept
            pred = float(np.clip(pred, 0.0, 100.0))
            predictions.append((d, round(pred, 2)))

    trend = _classify_trend(slope)
    risk_level = _get_risk_level(current_lvl, trend)

    # Build summary message
    trend_rate = abs(slope)
    if trend == "Increasing":
        msg = (
            f"Disease severity is increasing at ~{trend_rate:.1f}% per day. "
            f"Current severity: {current_sev:.1f}% ({current_lvl}). "
            f"Risk level: {risk_level}."
        )
    elif trend == "Decreasing":
        msg = (
            f"Disease severity is decreasing at ~{trend_rate:.1f}% per day. "
            f"Current severity: {current_sev:.1f}% ({current_lvl}). "
            f"Risk level: {risk_level}."
        )
    else:
        msg = (
            f"Disease severity is stable. "
            f"Current severity: {current_sev:.1f}% ({current_lvl}). "
            f"Risk level: {risk_level}."
        )

    return ForecastResult(
        current_severity=round(current_sev, 2),
        current_level=current_lvl,
        trend=trend,
        forecast=predictions,
        risk_level=risk_level,
        method=used_method,
        reliable=True,
        message=msg,
    )
