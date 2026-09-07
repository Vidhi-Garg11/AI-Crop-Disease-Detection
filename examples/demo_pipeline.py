"""
End-to-End Demo Pipeline
==========================

Demonstrates the complete XAI + Forecasting + Recommendation pipeline
using the existing MobileNetViTHybrid model and mock upstream outputs.

This script:
1. Loads the trained hybrid model from checkpoints/
2. Runs inference on a test image
3. Generates Grad-CAM explainability visualization
4. Computes severity from a mock mask (since no real segmentation is available)
5. Runs disease progression forecasting with mock historical data
6. Generates treatment recommendations
7. Produces a unified decision-support JSON output

All mock components are clearly labeled and will be replaced when the
real upstream modules from other team members are integrated.

Usage:
    cd AI-Crop-Disease-Detection
    python examples/demo_pipeline.py

    # Or with custom paths:
    python examples/demo_pipeline.py --checkpoint checkpoints/hybrid_model_best.pth --image data/test_image.JPG

Author: Vidhi Garg
"""

import argparse
import json
import os
import sys

# ==============================================================================
# Path setup — ensure src/ is importable
# ==============================================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
sys.path.insert(0, SRC_DIR)

import numpy as np
import torch
from PIL import Image

# Import XAI module components
from xai_module.gradcam import GradCAM, get_target_layer
from xai_module.severity_utils import build_severity_result
from xai_module.disease_forecasting import forecast_severity
from xai_module.treatment_recommender import generate_recommendation
from xai_module.decision_support import generate_decision_support_output, print_summary
from xai_module.mock_utils import (
    create_mock_classification_output,
    create_mock_severity_output,
    create_mock_severity_history,
    get_inference_preprocess,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="XAI Pipeline Demo — Crop Disease Decision Support"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=os.path.join(PROJECT_ROOT, "checkpoints", "hybrid_model_best.pth"),
        help="Path to the trained model checkpoint.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=os.path.join(PROJECT_ROOT, "data", "test_image.JPG"),
        help="Path to the input leaf image.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(PROJECT_ROOT, "outputs"),
        help="Directory to save output files.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # ==================================================================
    # Setup
    # ==================================================================
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    # ==================================================================
    # Step 1: Load the trained model
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 1: Loading Classification Model")
    print("=" * 60)

    if not os.path.exists(args.checkpoint):
        print(f"[ERROR] Checkpoint not found: {args.checkpoint}")
        print("        Please ensure the trained model checkpoint is available.")
        print("        Expected format: PyTorch checkpoint with 'model_state_dict' and 'classes'.")
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    classes = checkpoint["classes"]
    print(f"[INFO] Loaded checkpoint with {len(classes)} classes.")

    # Import and instantiate the hybrid model
    from hybrid_model import MobileNetViTHybrid

    model = MobileNetViTHybrid(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print("[INFO] Model loaded and set to eval mode.")

    # ==================================================================
    # Step 2: Run classification (using mock utility for now)
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 2: Classification [MOCK — uses existing model inference]")
    print("=" * 60)

    if not os.path.exists(args.image):
        print(f"[ERROR] Test image not found: {args.image}")
        print("        Please place a leaf image at the expected path.")
        sys.exit(1)

    # Get the preprocessing pipeline matching the trained model
    preprocess = get_inference_preprocess()

    classification_output = create_mock_classification_output(
        model=model,
        image_path=args.image,
        classes=classes,
        device=device,
        preprocess_fn=preprocess,
    )

    predicted_class = classification_output["predicted_class"]
    confidence = classification_output["confidence"]
    input_tensor = classification_output["input_tensor"]

    print(f"[INFO] Predicted class: {predicted_class}")
    print(f"[INFO] Confidence:      {confidence:.4f}")
    print(f"[INFO] Source:          {classification_output['source']}")

    # ==================================================================
    # Step 3: Grad-CAM Explainability
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 3: Grad-CAM Explainability")
    print("=" * 60)

    # Auto-detect the target convolutional layer
    target_layer = get_target_layer(model)
    print(f"[INFO] Target layer: {target_layer.__class__.__name__}")

    gradcam = GradCAM(model, target_layer)

    # Generate heatmap
    heatmap = gradcam.generate(input_tensor)
    print(f"[INFO] Heatmap shape: {heatmap.shape}, range: [{heatmap.min():.3f}, {heatmap.max():.3f}]")

    # Load original image for overlay
    original_image = Image.open(args.image).convert("RGB")
    overlay = gradcam.overlay(heatmap, original_image)

    # Save visualizations
    gradcam_paths = GradCAM.save_visualization(
        original_image=original_image,
        heatmap=heatmap,
        overlay=overlay,
        save_dir=args.output_dir,
        prefix="gradcam",
    )

    gradcam.remove_hooks()

    print(f"[INFO] Saved Grad-CAM visualizations:")
    for name, path in gradcam_paths.items():
        print(f"       {name}: {path}")

    # ==================================================================
    # Step 4: Severity Estimation [MOCK]
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 4: Severity Estimation [MOCK — using synthetic mask]")
    print("=" * 60)

    # Generate mock severity data
    mock_severity = create_mock_severity_output(image_shape=(224, 224), seed=42)
    print(f"[INFO] Source: {mock_severity['source']}")

    # Build severity result using the mock data
    severity_result = build_severity_result(
        severity_percentage=mock_severity["severity_percentage"],
        infected_mask=mock_severity["severity_mask"],
        leaf_mask=mock_severity["leaf_mask"],
        source="mock",
    )

    print(f"[INFO] Severity: {severity_result.percentage:.2f}% ({severity_result.level})")
    print(f"[INFO] Source:   {severity_result.source}")

    # ==================================================================
    # Step 5: Disease Progression Forecasting [MOCK HISTORY]
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 5: Disease Progression Forecasting [MOCK historical data]")
    print("=" * 60)

    # Generate mock historical severity data
    mock_history = create_mock_severity_history(pattern="increasing", seed=42)
    print(f"[INFO] Mock history: {mock_history}")

    forecast_result = forecast_severity(
        history=mock_history,
        future_days=7,
        method="linear",
    )

    print(f"[INFO] Trend:      {forecast_result.trend}")
    print(f"[INFO] Risk level: {forecast_result.risk_level}")
    print(f"[INFO] Reliable:   {forecast_result.reliable}")
    print(f"[INFO] Message:    {forecast_result.message}")
    if forecast_result.forecast:
        print("[INFO] Predictions:")
        for day, sev in forecast_result.forecast:
            print(f"       Day {day:3d} -> {sev:.1f}%")

    # ==================================================================
    # Step 6: Treatment Recommendation
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 6: Treatment Recommendation")
    print("=" * 60)

    recommendation = generate_recommendation(
        disease_class=predicted_class,
        severity_percentage=severity_result.percentage,
        severity_level=severity_result.level,
        trend=forecast_result.trend,
        confidence=confidence,
    )

    print(f"[INFO] Crop:           {recommendation.crop}")
    print(f"[INFO] Recommendation: {recommendation.recommendation}")
    print(f"[INFO] Monitoring:     {recommendation.monitoring}")
    print(f"[INFO] Warning:        {recommendation.warning}")

    # ==================================================================
    # Step 7: Unified Decision-Support Output
    # ==================================================================
    print("\n" + "=" * 60)
    print("  STEP 7: Generating Unified Decision-Support Output")
    print("=" * 60)

    final_output = generate_decision_support_output(
        # Classification
        predicted_class=predicted_class,
        confidence=confidence,
        # Severity
        severity_percentage=severity_result.percentage,
        severity_level=severity_result.level,
        severity_source=severity_result.source,
        # Grad-CAM
        gradcam_paths=gradcam_paths,
        # Forecast
        forecast_trend=forecast_result.trend,
        forecast_predictions=forecast_result.forecast,
        forecast_risk_level=forecast_result.risk_level,
        forecast_method=forecast_result.method,
        forecast_reliable=forecast_result.reliable,
        forecast_message=forecast_result.message,
        # Recommendation
        recommendation_action=recommendation.recommendation,
        recommendation_monitoring=recommendation.monitoring,
        # Metadata
        image_path=args.image,
        save_json=True,
        output_dir=args.output_dir,
    )

    # Print human-readable summary
    print_summary(final_output)

    print("[INFO] Pipeline complete! Check the outputs/ directory for results.")


if __name__ == "__main__":
    main()
