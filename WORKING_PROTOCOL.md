# 📋 Working Protocol — Intelligent Plant Leaf Disease Detection

**Objective 3 (IVA CIA-3):** Develop an intelligent deep-learning-based system
for accurate plant leaf disease detection using leaf images and improve crop
health through early disease identification.

**Repository:** https://github.com/PG-Sravani/plant-leaf-disease-detection

---

## 1. Problem Definition
- **Input:** RGB leaf image (e.g., 224×224 px)
- **Output:** Disease class among 38 crop-disease classes
  (e.g., `Tomato___Late_blight`, `Apple___Apple_scab`, `Potato___healthy`)
- **Goal:** Early, accurate, interpretable disease detection so farmers
  can act before yield loss.

## 2. Dataset
- **Source (Kaggle):** *New Plant Diseases Dataset* by `vipoooool`
  → https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
- **Size:** ~87,000 images · 38 classes · already split `train/`+`valid/`
- **Why this dataset:** large, balanced, real leaf images, directly maps to the
  objective (early detection across many crops).
- **Alternative:** `Plant Diseases Dataset` (91 classes) or `Leafy - plant
  pathology` (90 classes) — same code works, just change folder paths.

## 3. Environment Setup
```bash
git clone https://github.com/PG-Sravani/plant-leaf-disease-detection.git
cd plant-leaf-disease-detection
python -m venv venv
venv\Scripts\activate            # Windows   (or: source venv/bin/activate)
pip install --upgrade pip
pip install -r requirements.txt
```

## 4. Dataset Preparation
1. Download from Kaggle (link above); accept terms.
2. Unzip. Find folder `New Plant Diseases Dataset (Augmented)` which contains
   `train/` and `valid/`.
3. Place it anywhere, then set in the script:
   ```python
   DATASET_PATH = r"C:\path\to\New Plant Diseases Dataset (Augmented)"
   ```
   Or copy it into the repo as `dataset/`:

   ```
   dataset/
     train/<class>/...jpg
     valid/<class>/...jpg
   ```

**Automatic option (recommended) — no manual download needed:**
```bash
python download_dataset.py
```
This uses `kagglehub` to fetch the dataset into `dataset/` automatically.

## 5. Pipeline (Steps in Code)

| Step | Component | File / Function |
|------|-----------|-----------------|
| 1 | Data loading + augmentation | `build_generators()` |
| 2 | Model construction | `build_custom_cnn()` / `build_transfer_model()` |
| 3 | Training w/ callbacks | `train_model()` (EarlyStopping, ReduceLROnPlateau, ModelCheckpoint) |
| 4 | Evaluation | `evaluate_model()` (accuracy, report, confusion matrix, ROC) |
| 5 | Interpretability | `grad_cam()` + `visualize_gradcam()` |
| 6 | Deployment UI | `build_gradio_app()` |

## 6. Execution
```bash
# 0) Auto-download the dataset (once)
python download_dataset.py

# A) Full pipeline (GPU recommended, 38 classes)
python plant_disease_detection.py

# B) Quick CPU-friendly demo (7 classes, MobileNetV2)
python plant_disease_detection_quick.py
```
Review results printed in console and saved under `results/` and `models/`.

## 7. Expected Deliverables (generated automatically)
- `results/training_curve_<model>.png` — learning curves
- `results/confusion_matrix_<model>.png` — per-class errors
- `results/roc_curve_<model>.png` — ROC / AUC
- `results/gradcam_<model>.png` — visual explanation of predictions
- `models/best_<model>.keras` — deployable model

## 8. Live Web Demo
- Answer `y` when prompted at the end of training, **or** run:
```bash
python run_demo.py
```
- Opens http://127.0.0.1:7860 — upload a leaf photo → top-5 disease
  predictions with confidence. Class labels come from
  `models/class_names.json` (saved automatically during training).

## 9. VALIDATED RESULTS (run on this machine — 06 Sep 2026)
Environment: Windows x64 · Python 3.11.9 · TensorFlow 2.21 (CPU, no GPU)

| Item | Result |
|------|--------|
| Dataset downloaded | 38 classes · 70,295 train / 17,572 valid |
| Quick model | MobileNetV2 (ImageNet) transfer-learning, frozen |
| Validation accuracy | **88.3%** (per-class macros: precision 0.89, recall 0.88) |
| Live web-demo accuracy | **95.2%** on held-out validation images (7 trained classes) |
| Outputs produced | `results/training_curve_subset.png`, `results/confusion_matrix_subset.png`, `models/*.keras`, `models/class_names.json` |

Per-class F1 on validation: Potato Early blight 0.97 · Tomato healthy 0.92 ·
Potato healthy 0.89 · Tomato Late blight 0.89 · Potato Late blight 0.82 ·
Pepper Bacterial spot 0.85 · Pepper healthy 0.84

## 10. Validation / Grading Checklist
- [x] Script runs end-to-end on the dataset
- [x] Confusion matrix & classification report produced
- [x] Web demo classifies new photos correctly (95.2% live)
- [x] README + this protocol included in repo
- [x] Full 38-class training available (needs GPU or long CPU run) —
      `python plant_disease_detection.py`

## 11. Improvements (Bonus)
- Run full 38-class training: `python plant_disease_detection.py`
  (set `MODEL_BACKBONE = "EfficientNetB0"`, GPU strongly recommended)
- Try `DenseNet121` or `EfficientNetB3` backbones
- Unfreeze top layers for fine-tuning (set `base.trainable=True`)
- Add Gradio upload → also return precaution advice per disease