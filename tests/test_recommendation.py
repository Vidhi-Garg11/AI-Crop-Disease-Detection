"""
Tests for Treatment Recommendation Module
============================================

Tests recommendation generation for various severity/trend combinations,
healthy class detection, low-confidence warnings, and disclaimer presence.
"""

import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xai_module.treatment_recommender import (
    RecommendationResult,
    generate_recommendation,
    _extract_crop_name,
)


class TestCropNameExtraction:
    """Tests for parsing crop names from PlantVillage labels."""

    def test_standard_format(self):
        assert _extract_crop_name("Tomato___Late_blight") == "Tomato"

    def test_with_parentheses(self):
        crop = _extract_crop_name("Corn_(maize)___Common_rust_")
        assert "Corn" in crop
        assert "maize" in crop

    def test_with_comma(self):
        crop = _extract_crop_name("Pepper,_bell___Bacterial_spot")
        assert "Pepper" in crop

    def test_no_separator(self):
        """Should handle labels without ___ separator."""
        crop = _extract_crop_name("UnknownLabel")
        assert crop == "UnknownLabel"


class TestRecommendationGeneration:
    """Tests for the main recommendation generator."""

    def test_disease_specific_recommendation(self):
        """Known diseases should use disease-specific rules."""
        result = generate_recommendation(
            disease_class="Tomato___Late_blight",
            severity_percentage=35.0,
            severity_level="Moderate",
            trend="Increasing",
            confidence=0.92,
        )

        assert isinstance(result, RecommendationResult)
        assert result.crop == "Tomato"
        assert result.disease == "Tomato___Late_blight"
        assert result.severity_level == "Moderate"
        assert len(result.recommendation) > 0

    def test_unknown_disease_uses_generic(self):
        """Unknown diseases should fall back to generic recommendations."""
        result = generate_recommendation(
            disease_class="SomeNewCrop___UnknownDisease",
            severity_percentage=50.0,
            severity_level="Moderate",
            trend="Stable",
            confidence=0.80,
        )

        assert len(result.recommendation) > 0
        assert result.crop == "SomeNewCrop"

    def test_healthy_class(self):
        """Healthy plants should get no-treatment recommendation."""
        result = generate_recommendation(
            disease_class="Tomato___healthy",
            severity_percentage=0.0,
            severity_level="Low",
            trend="Stable",
            confidence=0.98,
        )

        assert "no disease" in result.recommendation.lower() or "regular monitoring" in result.recommendation.lower()

    def test_disclaimer_always_present(self):
        """Every recommendation must include the AI disclaimer."""
        result = generate_recommendation(
            disease_class="Potato___Late_blight",
            severity_percentage=70.0,
            severity_level="Severe",
            trend="Increasing",
            confidence=0.90,
        )

        assert "consult" in result.warning.lower()
        assert "expert" in result.warning.lower()

    def test_low_confidence_warning(self):
        """Low confidence should trigger an additional note."""
        result = generate_recommendation(
            disease_class="Tomato___Late_blight",
            severity_percentage=20.0,
            severity_level="Mild",
            trend="Stable",
            confidence=0.45,
        )

        assert "confidence" in result.recommendation.lower()

    def test_high_confidence_no_extra_warning(self):
        """High confidence should not trigger extra note."""
        result = generate_recommendation(
            disease_class="Tomato___Late_blight",
            severity_percentage=20.0,
            severity_level="Mild",
            trend="Stable",
            confidence=0.95,
        )

        # Should not contain the low-confidence caveat
        assert "confidence is low" not in result.recommendation.lower()


class TestRecommendationResult:
    """Tests for the result structure."""

    def test_to_dict(self):
        """to_dict should return a JSON-serializable dictionary."""
        result = generate_recommendation(
            disease_class="Apple___Apple_scab",
            severity_percentage=15.0,
            severity_level="Mild",
            trend="Increasing",
            confidence=0.88,
        )

        d = result.to_dict()
        assert isinstance(d, dict)
        assert "crop" in d
        assert "disease" in d
        assert "confidence" in d
        assert "recommendation" in d
        assert "warning" in d

    def test_all_severity_levels(self):
        """Should generate valid recommendations for all severity levels."""
        for level in ["Low", "Mild", "Moderate", "Severe"]:
            result = generate_recommendation(
                disease_class="Tomato___Early_blight",
                severity_percentage=50.0,
                severity_level=level,
                trend="Stable",
                confidence=0.85,
            )
            assert len(result.recommendation) > 10  # Non-trivial text

    def test_progression_modifier_applied(self):
        """Increasing trend should add urgency modifier."""
        result = generate_recommendation(
            disease_class="Potato___Late_blight",
            severity_percentage=40.0,
            severity_level="Moderate",
            trend="Increasing",
            confidence=0.90,
        )

        # The progression modifier mentions "spreading" or "promptly"
        assert "spreading" in result.recommendation.lower() or "promptly" in result.recommendation.lower()
