# AI-Based Crop Disease Detection and Health Monitoring System

Using Image Processing and Deep Learning for automated plant disease identification, explainability, severity assessment, progression forecasting, and treatment recommendations.

## Project Pipeline

```
Image Input
    ↓
Image Preprocessing (BilateralFilter + VegetationIndices)
    ↓
Disease Localization / Segmentation (Aryaahi)
    ↓
Disease Classification (Sravani / Dominic)
    ↓
Severity Estimation (Aryaahi)
    ↓
XAI / Grad-CAM (Vidhi)
    ↓
Disease Progression Forecasting (Vidhi)
    ↓
Treatment Recommendation (Vidhi)
    ↓
Final Decision-Support Output (Vidhi)
```

## Team Members

| Member | Module | Location |
|--------|--------|----------|
| **Sravani** | Classification model | `src/` |
| **Dominic** | CNN-ViT hybrid classification model | `src/hybrid_model.py` |
| **Aryaahi** | Leaf/disease segmentation & severity mask | TBD |
| **Vidhi** | XAI + Forecasting + Recommendation | `src/xai_module/` |

---

## Repository Structure

```
AI-Crop-Disease-Detection/
├── src/
│   ├── hybrid_model.py              # MobileNetV3-ViT hybrid classifier (Dominic)
│   ├── data_loader.py               # PlantVillage dataset loader
│   ├── train.py                     # Training script
│   ├── inference.py                 # Single-image inference
│   ├── evaluate_metrics.py          # Validation metrics
│   └── xai_module/                  # Vidhi's XAI module
│       ├── __init__.py
│       ├── config.py                # All configurable thresholds & rules
│       ├── gradcam.py               # Grad-CAM explainability engine
│       ├── severity_utils.py        # Severity calculation & categorization
│       ├── disease_forecasting.py   # Progression forecasting
│       ├── treatment_recommender.py # Treatment recommendation engine
│       ├── decision_support.py      # Unified output integrator
│       └── mock_utils.py            # Mock data generators [MOCK]
├── tests/
│   ├── test_gradcam.py
│   ├── test_severity.py
│   ├── test_forecasting.py
│   └── test_recommendation.py
├── examples/
│   └── demo_pipeline.py             # End-to-end demo script
├── checkpoints/                     # Model weights (gitignored)
├── data/                            # Dataset (gitignored)
│   └── raw/color/                   # PlantVillage images
├── outputs/                         # Generated outputs (gitignored)
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and switch to the feature branch

```bash
git clone https://github.com/Vidhi-Garg11/AI-Crop-Disease-Detection.git
cd AI-Crop-Disease-Detection
git checkout feature/vidhi-xai
```

### 2. Create a virtual environment (recommended)

```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Dataset setup

Download the PlantVillage dataset from Kaggle:
- **URL**: https://www.kaggle.com/datasets/siddhantsadangi/plantvillagedataset?select=raw
- Extract into `data/raw/color/` so the structure is:
  ```
  data/raw/color/
  ├── Apple___Apple_scab/
  ├── Apple___Black_rot/
  ├── Tomato___Late_blight/
  └── ... (38 class directories)
  ```
- **Do NOT** commit the dataset to Git (it is gitignored).

### 5. Model checkpoint

Ensure the trained model checkpoint exists at:
```
checkpoints/hybrid_model_best.pth
```

This checkpoint is produced by `src/train.py` and contains `model_state_dict` and `classes`.

---

## Running the XAI Pipeline

### Full demo (end-to-end pipeline)

```bash
python examples/demo_pipeline.py
```

With custom paths:

```bash
python examples/demo_pipeline.py --checkpoint checkpoints/hybrid_model_best.pth --image data/test_image.JPG --output-dir outputs
```

### Run tests

```bash
python -m pytest tests/ -v
```

---

## XAI Module Components

### 1. Grad-CAM (`src/xai_module/gradcam.py`)

Architecture-agnostic Grad-CAM implementation that works with ResNet, EfficientNet, MobileNet, and the project's MobileNetViTHybrid.

```python
from xai_module.gradcam import GradCAM, get_target_layer

# Auto-detect target layer
target_layer = get_target_layer(model)

# Generate heatmap
gradcam = GradCAM(model, target_layer)
heatmap = gradcam.generate(input_tensor, target_class=predicted_idx)
overlay = gradcam.overlay(heatmap, original_image)
gradcam.remove_hooks()
```

### 2. Severity Integration (`src/xai_module/severity_utils.py`)

Calculates disease severity from segmentation masks or accepts pre-computed percentages.

```python
from xai_module.severity_utils import build_severity_result

# From upstream segmentation module
result = build_severity_result(
    severity_percentage=37.5,  # OR infected_mask=mask
)
# → SeverityResult(percentage=37.5, level="Moderate", source="direct")
```

### 3. Disease Forecasting (`src/xai_module/disease_forecasting.py`)

Predicts future severity from historical observations using linear or exponential models.

```python
from xai_module.disease_forecasting import forecast_severity

history = [(1, 10), (3, 18), (5, 31), (7, 45)]
result = forecast_severity(history, future_days=7, method="linear")
# → ForecastResult(trend="Increasing", forecast=[(8, 52.3), ...], ...)
```

### 4. Treatment Recommendation (`src/xai_module/treatment_recommender.py`)

Rule-based recommendation engine with disease-specific and generic fallback rules.

```python
from xai_module.treatment_recommender import generate_recommendation

rec = generate_recommendation(
    disease_class="Tomato___Late_blight",
    severity_percentage=37.5,
    severity_level="Moderate",
    trend="Increasing",
    confidence=0.94,
)
```

### 5. Decision Support Output (`src/xai_module/decision_support.py`)

Combines all pipeline outputs into a single structured JSON result.

### 6. Configuration (`src/xai_module/config.py`)

All configurable values in one file:
- Severity thresholds (0–10% Low, 10–30% Mild, 30–60% Moderate, 60–100% Severe)
- Treatment rules per disease
- Forecasting parameters
- Monitoring recommendations
- Disclaimer text

---

## Integration Guide for Team Members

### For Sravani & Dominic (Classification)

Your classification model should produce:

```python
classification_output = {
    "predicted_class": "Tomato___Late_blight",  # PlantVillage class label
    "confidence": 0.94,                         # float, 0–1
    "model": trained_pytorch_model,              # nn.Module in eval mode
}
```

The XAI module will use `model` for Grad-CAM and `predicted_class`/`confidence` for downstream processing.

### For Aryaahi (Segmentation / Severity)

Your segmentation module should produce:

```python
severity_output = {
    "severity_mask": np.ndarray,          # binary mask (H, W), 1 = infected
    "leaf_mask": np.ndarray or None,      # optional leaf-region mask
    "severity_percentage": float,         # pre-computed severity %
}
```

Either `severity_mask` or `severity_percentage` can be provided. If both are given, the pre-computed percentage is used directly.

### Replacing mock components

When real upstream modules are ready:

1. Replace calls to `mock_utils.create_mock_classification_output()` with the real classification output.
2. Replace calls to `mock_utils.create_mock_severity_output()` with the real segmentation output.
3. Replace calls to `mock_utils.create_mock_severity_history()` with real temporal severity data (if available).

All mock functions are in `src/xai_module/mock_utils.py` and clearly labeled with `[MOCK]`.

---

## Example Output

The pipeline produces a JSON decision-support file like:

```json
{
  "prediction": {
    "disease": "Tomato___Late_blight",
    "confidence": 0.94
  },
  "severity": {
    "percentage": 37.5,
    "level": "Moderate",
    "source": "mock"
  },
  "explainability": {
    "method": "Grad-CAM",
    "gradcam_overlay": "outputs/gradcam_overlay.png"
  },
  "forecast": {
    "trend": "Increasing",
    "risk_level": "High",
    "future_severity": [
      {"day": 8, "predicted_severity": 52.3},
      {"day": 9, "predicted_severity": 58.7}
    ]
  },
  "recommendation": {
    "action": "Isolate severely affected plants...",
    "monitoring": "Inspect plants daily...",
    "warning": "AI-generated recommendation; consult an agricultural expert..."
  }
}
```

---

## Important Notes

- **Severity thresholds** are project-defined defaults, NOT agriculturally validated. They can be adjusted in `config.py`.
- **Treatment recommendations** are high-level best practices. No specific chemical dosages are provided.
- **Mock data** is clearly labeled `[MOCK]` throughout. Do not use mock outputs in production.
- **PlantVillage** does not provide severity annotations, time-series data, or segmentation masks. All such data used during development is synthetic.

---

## License

This project is for academic purposes as part of the AI-Based Crop Disease Detection course project.