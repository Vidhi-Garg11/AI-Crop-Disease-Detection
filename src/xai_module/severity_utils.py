"""
Severity Calculation & Categorization
=======================================

Provides functions to:
- Calculate disease severity percentage from segmentation masks
- Map severity percentages to interpretable categories (Low/Mild/Moderate/Severe)

Designed to consume the output of the upstream segmentation module (Aryaahi).
If no real segmentation mask is available, severity_percentage can be supplied directly.

NOTE: The PlantVillage dataset does NOT contain ground-truth severity masks or
severity percentages. During development, use mock_utils.py for test data.

Author: Vidhi Garg
"""

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np

from .config import SEVERITY_THRESHOLDS


@dataclass
class SeverityResult:
    """
    Structured container for severity assessment output.

    Attributes:
        percentage: Disease severity as a percentage (0–100).
        level: Human-readable severity category (e.g., "Low", "Moderate").
        source: How the severity was determined:
                "mask" — computed from segmentation masks.
                "direct" — provided as a pre-computed percentage.
                "mock" — generated from mock/test data.
    """
    percentage: float
    level: str
    source: str


def calculate_severity(
    infected_mask: np.ndarray,
    leaf_mask: Optional[np.ndarray] = None,
) -> float:
    """
    Calculate disease severity as a percentage of infected leaf area.

    severity_percentage = (infected_leaf_area / total_leaf_area) * 100

    Args:
        infected_mask: Binary mask (H, W) where 1/True = infected pixel.
                       Can be integer or boolean dtype.
        leaf_mask: Optional binary mask (H, W) where 1/True = leaf pixel.
                   If None, the entire image area is treated as leaf area.

                   ASSUMPTION: When leaf_mask is not provided, the calculation
                   assumes ALL pixels in the image belong to the leaf. This
                   will underestimate severity if there is significant background.
                   For accurate results, provide a leaf segmentation mask.

    Returns:
        Severity percentage (0.0 to 100.0).

    Raises:
        ValueError: If masks have incompatible shapes or zero leaf area.
    """
    if infected_mask.ndim != 2:
        raise ValueError(
            f"infected_mask must be 2D (H, W), got shape {infected_mask.shape}"
        )

    if leaf_mask is not None:
        if leaf_mask.ndim != 2:
            raise ValueError(
                f"leaf_mask must be 2D (H, W), got shape {leaf_mask.shape}"
            )
        if infected_mask.shape != leaf_mask.shape:
            raise ValueError(
                f"Mask shape mismatch: infected_mask {infected_mask.shape} "
                f"vs leaf_mask {leaf_mask.shape}"
            )

    # Convert to boolean for counting
    infected = infected_mask.astype(bool)

    if leaf_mask is not None:
        leaf = leaf_mask.astype(bool)
        # Only count infected pixels that are within the leaf region
        infected_area = np.logical_and(infected, leaf).sum()
        total_area = leaf.sum()
    else:
        infected_area = infected.sum()
        total_area = infected_mask.size  # entire image

    if total_area == 0:
        raise ValueError(
            "Total leaf area is zero. Cannot compute severity. "
            "Check that the leaf_mask is not empty."
        )

    severity_pct = (infected_area / total_area) * 100.0

    # Clamp to [0, 100]
    return float(np.clip(severity_pct, 0.0, 100.0))


def get_severity_level(percentage: float) -> str:
    """
    Map a severity percentage to a human-readable category.

    Uses thresholds defined in config.SEVERITY_THRESHOLDS.

    Default mapping:
        0–10%   → "Low"
        10–30%  → "Mild"
        30–60%  → "Moderate"
        60–100% → "Severe"

    Args:
        percentage: Severity percentage (0.0 to 100.0).

    Returns:
        Category string (e.g., "Low", "Mild", "Moderate", "Severe").
    """
    if percentage < 0.0 or percentage > 100.0:
        raise ValueError(
            f"Severity percentage must be in [0, 100], got {percentage}"
        )

    for upper_bound, label in SEVERITY_THRESHOLDS:
        if percentage < upper_bound:
            return label

    # Fallback (should not be reached with proper thresholds)
    return SEVERITY_THRESHOLDS[-1][1]


def build_severity_result(
    severity_percentage: Optional[float] = None,
    infected_mask: Optional[np.ndarray] = None,
    leaf_mask: Optional[np.ndarray] = None,
    source: Optional[str] = None,
) -> SeverityResult:
    """
    Build a SeverityResult from either a direct percentage or masks.

    Priority:
        1. If severity_percentage is provided, use it directly.
        2. If infected_mask is provided, compute severity from masks.
        3. If neither, raise an error.

    This matches the expected upstream segmentation output format:
        {
            "severity_mask": np.ndarray or None,
            "severity_percentage": float or None
        }

    Args:
        severity_percentage: Pre-computed severity percentage (from upstream).
        infected_mask: Binary infected-area mask.
        leaf_mask: Optional binary leaf-area mask.
        source: Override for the source label.

    Returns:
        SeverityResult with percentage, level, and source.
    """
    if severity_percentage is not None:
        pct = float(np.clip(severity_percentage, 0.0, 100.0))
        return SeverityResult(
            percentage=round(pct, 2),
            level=get_severity_level(pct),
            source=source or "direct",
        )
    elif infected_mask is not None:
        pct = calculate_severity(infected_mask, leaf_mask)
        return SeverityResult(
            percentage=round(pct, 2),
            level=get_severity_level(pct),
            source=source or "mask",
        )
    else:
        raise ValueError(
            "Must provide either severity_percentage or infected_mask. "
            "Neither was given."
        )
