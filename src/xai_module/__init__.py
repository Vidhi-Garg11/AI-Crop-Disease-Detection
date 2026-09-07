"""
Vidhi XAI Module — Explainable AI, Forecasting & Treatment Recommendation
==========================================================================

This package provides:
- Grad-CAM explainability for plant disease classification models
- Disease severity calculation and categorization
- Disease progression forecasting
- Treatment recommendation engine
- Unified decision-support output integrator

Author: Vidhi Garg
Module: XAI + Forecasting + Recommendation
"""

from .gradcam import GradCAM, get_target_layer
from .severity_utils import calculate_severity, get_severity_level, SeverityResult
from .disease_forecasting import forecast_severity, ForecastResult
from .treatment_recommender import generate_recommendation, RecommendationResult
from .decision_support import generate_decision_support_output

__all__ = [
    "GradCAM",
    "get_target_layer",
    "calculate_severity",
    "get_severity_level",
    "SeverityResult",
    "forecast_severity",
    "ForecastResult",
    "generate_recommendation",
    "RecommendationResult",
    "generate_decision_support_output",
]
