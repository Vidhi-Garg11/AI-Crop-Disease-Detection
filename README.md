# 🌿 Intelligent Plant Leaf Disease Detection

**Course:** Image and Video Analytics (IVA) — CIA 3
**Objective 3:** Develop an intelligent deep learning-based system for accurate
plant leaf disease detection using leaf images and improve crop health through
early disease identification.

---

## 📌 Objective Fulfilment

This project delivers a complete deep-learning pipeline that:

1. **Ingests leaf images** of crops from multiple species.
2. **Detects & classifies diseases** (bacterial, fungal, viral) vs. healthy leaves.
3. **Explains predictions** with Grad-CAM heatmaps (interpretable AI).
4. **Exposes a real-world web interface** (Gradio) for farmer/usability testing.
5. **Reports** accuracy, confusion matrix, ROC curves, and per-class metrics.

---

## 📊 Recommended Dataset (Kaggle)

**Primary (full, 38 classes):**
- **New Plant Diseases Dataset** — *vipoooool*
- https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- Derived from the PlantVillage dataset. ~87k images across 38 crop-disease
  classes, already split into `train` / `valid`.

**Optional test images:**
- **Plant Disease Detection - Test Set Images**
- https://www.kaggle.com/datasets/emmarex/plantdisease

**How to set it up:**
1. Create a Kaggle account and accept the dataset terms.
2. Download and unzip. Locate the folder
   `New Plant Diseases Dataset (Augmented)` which contains `train` and `valid`.
3. Place it inside this project under `dataset/`, or point the script to it:

   ```python
   DATASET_PATH = r"C:\path\to\New Plant Diseases Dataset (Augmented)"
   ```

Expected tree:

```
dataset/
  train/
    Apple___Apple_scab/
    Apple___Black_rot/
    ...
  valid/
    ...
```

### ⚡ Automatic download (recommended)

```bash
pip install kagglehub          # already in requirements.txt
python download_dataset.py
```

This downloads the dataset via `kagglehub`, copies `train/` + `valid/` straight
into `dataset/`, and verifies class/image counts. No manual unzip needed.
(A Kaggle login may be requested on first use.)

---

## 🚀 Installation

```bash
# Create & activate a virtual environment (recommended)
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

If you have no GPU and would like faster installs,
`pip install tensorflow-cpu` is also an option.

---

## ▶️ How to Run

### Option A — Full training (38 classes, needs a GPU)

```bash
python plant_disease_detection.py
```

Prompts at the end let you also launch the Gradio web app.

### Option B — Quick training (subclassed subset, runs on CPU/laptop)

Edit `SUBSET_CLASSES` in `plant_disease_detection_quick.py`, then:

```bash
python plant_disease_detection_quick.py
```

This trains a transfer-learning model on a handful of classes in minutes.

### Option C — Real-time Web Demo

```bash
python run_demo.py          # loads trained model, opens http://127.0.0.1:7860
```

Or answer `y` at the end of the training scripts to launch the demo
automatically.

### Option D — Hybrid-backend Web Demo (MobileNetV3 + Vision Transformer)

A second, more advanced backend is included, adapted from
[Vidhi-Garg11/AI-Crop-Disease-Detection](https://github.com/Vidhi-Garg11/AI-Crop-Disease-Detection).
It serves predictions from a pretrained **25-class** PyTorch checkpoint
(`hybrid/hybrid_model_best.pth`) using the same style of upload UI.

```bash
# one-time setup (CPU wheels)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

python run_demo_hybrid.py   # opens http://127.0.0.1:7860
```

Model: MobileNetV3-Large CNN → 1×1 projection → positional embeddings →
2 transformer encoder blocks → classification head (3.7M params).
Preprocessing: bilateral filter → vegetation indices (ExG/ExR) →
ImageNet normalization. See `hybrid/hybrid_backend.py`.

---

## 🧠 Models Included

| Backbone      | Type                 | Speed/Resource | Accuracy (typical) |
|---------------|----------------------|----------------|--------------------|
| Custom CNN    | From scratch         | Fast, light    | ~70–85%           |
| MobileNetV2   | Transfer learning    | Fast, light    | ~95%+             |
| EfficientNetB0| Transfer learning    | Medium         | ~96%+             |
| VGG16         | Transfer learning    | Heavy          | ~95%+             |
| ResNet50      | Transfer learning    | Heavy          | ~96%+             |
| DenseNet121   | Transfer learning    | Heavy          | ~97%+             |

Change `MODEL_BACKBONE` at the top of `plant_disease_detection.py`.

---

## 📁 Outputs (saved in `results/` and `models/`)

```
results/
  training_curve_<model>.png
  confusion_matrix_<model>.png
  roc_curve_<model>.png
  gradcam_<model>.png
models/
  best_<model>.keras
  plant_disease_<model>_final.keras
```

---

## 🔬 Features Delivered

- **Image augmentation** (rotation, flips, zoom, brightness) — improves
  generalization to real field conditions.
- **Early disease identification** — model classifies before symptoms get
  severe, enabling timely intervention.
- **Grad-CAM interpretability** — shows *where* in the leaf the model sees
  disease (builds trust / supports agronomists).
- **Full evaluation** — accuracy, classification report, confusion matrix,
  ROC curves.
- **Web UI** — farmers/agronomists upload a leaf photo and get a diagnosis.

---

## ⚠️ Notes

- Training on the **full 38-class dataset** with a heavy backbone requires a GPU
  (comments in code split by resource). The quick variant is CPU-friendly.
- Dataset is CC BY / research licensed — cite PlantVillage and the Kaggle
  author if you publish.
- Model predictions aid diagnosis but should be confirmed by agronomists.

---

## ✍️ Author

Built for IVA CIA-3 (Objective 3).
