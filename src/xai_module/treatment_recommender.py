"""
Treatment Recommendation Engine
=================================

Rule-based recommendation system that generates structured agricultural
decision-support recommendations based on disease diagnosis, severity,
and progression trends.

All recommendations are high-level agricultural best practices.
No specific chemical dosages or unsafe pesticide instructions are included.

Recommendations are loaded from config.py and can be easily modified
by the team without changing this module's logic.

Author: Vidhi Garg
"""

from dataclasses import dataclass
from typing import Optional

from .config import (
    DISCLAIMER,
    GENERIC_RECOMMENDATIONS,
    MONITORING_RECOMMENDATIONS,
    PROGRESSION_MODIFIERS,
    TREATMENT_RULES,
)


@dataclass
class RecommendationResult:
    """
    Structured treatment recommendation output.

    Attributes:
        crop: Crop name extracted from the disease class label.
        disease: Full disease class label (PlantVillage format).
        confidence: Classification model confidence score.
        severity_percentage: Disease severity percentage.
        severity_level: Severity category (Low/Mild/Moderate/Severe).
        progression: Trend direction (Increasing/Stable/Decreasing/Unknown).
        recommendation: Primary treatment/action recommendation.
        monitoring: Monitoring schedule recommendation.
        warning: Standard AI disclaimer.
    """
    crop: str
    disease: str
    confidence: float
    severity_percentage: float
    severity_level: str
    progression: str
    recommendation: str
    monitoring: str
    warning: str

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "crop": self.crop,
            "disease": self.disease,
            "confidence": self.confidence,
            "severity_percentage": self.severity_percentage,
            "severity_level": self.severity_level,
            "progression": self.progression,
            "recommendation": self.recommendation,
            "monitoring": self.monitoring,
            "warning": self.warning,
        }


def _extract_crop_name(disease_class: str) -> str:
    """
    Extract the crop name from a PlantVillage class label.

    PlantVillage format: "Crop___Disease" (triple underscore separator).
    Example: "Tomato___Late_blight" → "Tomato"
             "Corn_(maize)___Common_rust_" → "Corn (maize)"

    Args:
        disease_class: PlantVillage class label string.

    Returns:
        Crop name, cleaned up for display.
    """
    if "___" in disease_class:
        crop = disease_class.split("___")[0]
    else:
        # Fallback: use the whole string
        crop = disease_class

    # Clean up underscores for display
    crop = crop.replace("_", " ").strip()
    # Fix double spaces
    while "  " in crop:
        crop = crop.replace("  ", " ")

    return crop


def _get_disease_recommendation(
    disease_class: str,
    severity_level: str,
) -> str:
    """
    Look up disease-specific recommendation from TREATMENT_RULES config.

    Falls back to GENERIC_RECOMMENDATIONS if the disease is not found
    or if the severity level is not covered.

    Args:
        disease_class: PlantVillage disease class label.
        severity_level: Severity category (Low/Mild/Moderate/Severe).

    Returns:
        Recommendation string.
    """
    # Check disease-specific rules first
    if disease_class in TREATMENT_RULES:
        disease_rules = TREATMENT_RULES[disease_class]
        if severity_level in disease_rules:
            return disease_rules[severity_level]

    # Fallback to generic recommendations
    return GENERIC_RECOMMENDATIONS.get(
        severity_level,
        GENERIC_RECOMMENDATIONS.get("Moderate", "Consult an agricultural expert."),
    )


def _get_monitoring_recommendation(severity_level: str) -> str:
    """Get monitoring schedule recommendation based on severity level."""
    return MONITORING_RECOMMENDATIONS.get(
        severity_level,
        "Monitor plants regularly and consult an expert if condition worsens.",
    )


def _apply_progression_modifier(
    recommendation: str,
    progression: str,
) -> str:
    """Append progression-based action modifier to the recommendation."""
    modifier = PROGRESSION_MODIFIERS.get(progression, "")
    if modifier:
        return recommendation + modifier
    return recommendation


def generate_recommendation(
    disease_class: str,
    severity_percentage: float,
    severity_level: str,
    trend: str = "Unknown",
    confidence: float = 0.0,
) -> RecommendationResult:
    """
    Generate a structured treatment recommendation.

    Args:
        disease_class: Predicted disease class (PlantVillage format, e.g., "Tomato___Late_blight").
        severity_percentage: Disease severity percentage (0–100).
        severity_level: Severity category (Low/Mild/Moderate/Severe).
        trend: Disease progression trend (Increasing/Stable/Decreasing/Unknown).
        confidence: Classification model confidence score (0–1).

    Returns:
        RecommendationResult with all recommendation fields populated.
    """
    crop = _extract_crop_name(disease_class)

    # Check if this is a "healthy" class — no treatment needed
    is_healthy = "healthy" in disease_class.lower()

    if is_healthy:
        recommendation = (
            "No disease detected. Continue regular monitoring and maintain "
            "good agricultural practices including proper irrigation, "
            "fertilization, and field hygiene."
        )
        monitoring = "Continue routine weekly inspections."
    else:
        # Get disease/severity-specific recommendation
        recommendation = _get_disease_recommendation(disease_class, severity_level)
        # Append progression modifier
        recommendation = _apply_progression_modifier(recommendation, trend)
        # Get monitoring schedule
        monitoring = _get_monitoring_recommendation(severity_level)

    # Add confidence caveat if confidence is low
    if confidence > 0 and confidence < 0.7:
        recommendation += (
            f" NOTE: Classification confidence is low ({confidence:.0%}). "
            "Consider obtaining a second opinion or additional testing."
        )

    return RecommendationResult(
        crop=crop,
        disease=disease_class,
        confidence=round(confidence, 4),
        severity_percentage=round(severity_percentage, 2),
        severity_level=severity_level,
        progression=trend,
        recommendation=recommendation,
        monitoring=monitoring,
        warning=DISCLAIMER,
    )
