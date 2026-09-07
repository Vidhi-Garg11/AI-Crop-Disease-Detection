"""
Unified Decision-Support Output Integrator
=============================================

Combines outputs from all pipeline stages into a single structured
JSON-serializable result for downstream consumption.

Pipeline stages integrated:
    1. Classification (from Sravani / Dominic)
    2. Severity estimation (from Aryaahi)
    3. Grad-CAM explainability (Vidhi)
    4. Disease progression forecasting (Vidhi)
    5. Treatment recommendation (Vidhi)

Author: Vidhi Garg
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .config import DEFAULT_OUTPUT_DIR, DISCLAIMER


def generate_decision_support_output(
    # --- Classification (upstream) ---
    predicted_class: str,
    confidence: float,
    # --- Severity (upstream or computed) ---
    severity_percentage: float,
    severity_level: str,
    severity_source: str = "unknown",
    # --- Grad-CAM (this module) ---
    gradcam_paths: Optional[Dict[str, str]] = None,
    # --- Forecast (this module) ---
    forecast_trend: str = "Unknown",
    forecast_predictions: Optional[List[Tuple[int, float]]] = None,
    forecast_risk_level: str = "Unknown",
    forecast_method: str = "N/A",
    forecast_reliable: bool = False,
    forecast_message: str = "",
    # --- Recommendation (this module) ---
    recommendation_action: str = "",
    recommendation_monitoring: str = "",
    # --- Metadata ---
    image_path: Optional[str] = None,
    save_json: bool = True,
    output_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Combine all pipeline outputs into one unified decision-support result.

    This function acts as the final integration point that downstream
    applications (dashboards, reports, APIs) can consume.

    Args:
        predicted_class: Disease class label from classification model.
        confidence: Classification confidence score (0–1).
        severity_percentage: Computed or received severity percentage.
        severity_level: Severity category string.
        severity_source: How severity was determined ("mask", "direct", "mock").
        gradcam_paths: Dict of saved Grad-CAM image paths
                       (keys: "original", "heatmap", "overlay").
        forecast_trend: Trend direction string.
        forecast_predictions: List of (day, severity) prediction tuples.
        forecast_risk_level: Risk assessment string.
        forecast_method: Forecasting method used.
        forecast_reliable: Whether forecast had sufficient data.
        forecast_message: Human-readable forecast summary.
        recommendation_action: Treatment recommendation text.
        recommendation_monitoring: Monitoring recommendation text.
        image_path: Path to the input image (for reference).
        save_json: If True, saves the output to a JSON file.
        output_dir: Directory to save the JSON output.

    Returns:
        JSON-serializable dictionary with the complete decision-support output.
    """
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "image_path": image_path,
            "pipeline_version": "1.0.0",
            "disclaimer": DISCLAIMER,
        },
        "prediction": {
            "disease": predicted_class,
            "confidence": round(confidence, 4),
        },
        "severity": {
            "percentage": round(severity_percentage, 2),
            "level": severity_level,
            "source": severity_source,
        },
        "explainability": {
            "method": "Grad-CAM",
            "gradcam_original": (
                gradcam_paths.get("original") if gradcam_paths else None
            ),
            "gradcam_heatmap": (
                gradcam_paths.get("heatmap") if gradcam_paths else None
            ),
            "gradcam_overlay": (
                gradcam_paths.get("overlay") if gradcam_paths else None
            ),
        },
        "forecast": {
            "trend": forecast_trend,
            "risk_level": forecast_risk_level,
            "method": forecast_method,
            "reliable": forecast_reliable,
            "future_severity": (
                [{"day": d, "predicted_severity": s} for d, s in forecast_predictions]
                if forecast_predictions
                else []
            ),
            "message": forecast_message,
        },
        "recommendation": {
            "action": recommendation_action,
            "monitoring": recommendation_monitoring,
            "warning": DISCLAIMER,
        },
    }

    # Save to JSON file if requested
    if save_json:
        if output_dir is None:
            output_dir = DEFAULT_OUTPUT_DIR
        os.makedirs(output_dir, exist_ok=True)

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = os.path.join(output_dir, f"decision_support_{timestamp_str}.json")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)

        output["metadata"]["output_file"] = json_path
        print(f"[INFO] Decision-support output saved to: {json_path}")

    return output


def print_summary(output: Dict[str, Any]) -> None:
    """
    Print a human-readable summary of the decision-support output.

    Args:
        output: The dictionary returned by generate_decision_support_output.
    """
    print("\n" + "=" * 60)
    print("  DECISION-SUPPORT REPORT")
    print("=" * 60)

    pred = output.get("prediction", {})
    print(f"\n[Prediction]")
    print(f"   Disease:    {pred.get('disease', 'N/A')}")
    print(f"   Confidence: {pred.get('confidence', 0):.2%}")

    sev = output.get("severity", {})
    print(f"\n[Severity]")
    print(f"   Percentage: {sev.get('percentage', 'N/A')}%")
    print(f"   Level:      {sev.get('level', 'N/A')}")
    print(f"   Source:     {sev.get('source', 'N/A')}")

    xai = output.get("explainability", {})
    print(f"\n[Explainability (Grad-CAM)]")
    if xai.get("gradcam_overlay"):
        print(f"   Overlay:  {xai['gradcam_overlay']}")
        print(f"   Heatmap:  {xai.get('gradcam_heatmap', 'N/A')}")
    else:
        print("   Not generated.")

    fc = output.get("forecast", {})
    print(f"\n[Forecast]")
    print(f"   Trend:      {fc.get('trend', 'N/A')}")
    print(f"   Risk Level: {fc.get('risk_level', 'N/A')}")
    print(f"   Reliable:   {fc.get('reliable', False)}")
    future = fc.get("future_severity", [])
    if future:
        print(f"   Predictions:")
        for entry in future[:5]:  # Show at most 5
            print(f"     Day {entry['day']:3d} -> {entry['predicted_severity']:.1f}%")
        if len(future) > 5:
            print(f"     ... and {len(future) - 5} more")
    print(f"   Message:    {fc.get('message', 'N/A')}")

    rec = output.get("recommendation", {})
    print(f"\n[Recommendation]")
    print(f"   Action:     {rec.get('action', 'N/A')}")
    print(f"   Monitoring: {rec.get('monitoring', 'N/A')}")

    print(f"\n[WARNING] {rec.get('warning', DISCLAIMER)}")
    print("=" * 60 + "\n")

