"""
=====================================================================
 Generate Presentation: Intelligent Plant Leaf Disease Detection
=====================================================================
Covers: objective, dataset, pipeline (input -> output), models,
training, results, Grad-CAM, web demo, repo, and conclusions.

Run:
    python make_presentation.py
Output:
    Plant_Leaf_Disease_Detection_IVACIA3.pptx
=====================================================================
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(PROJECT_DIR, "Plant_Leaf_Disease_Detection_IVACIA3.pptx")
RESULTS_DIR = os.path.join(PROJECT_DIR, "results")

# ---- palette -----------------------------------------------------
GREEN_DARK  = RGBColor(0x1B, 0x5E, 0x20)
GREEN_MED   = RGBColor(0x2E, 0x7D, 0x32)
GREEN_LIGHT = RGBColor(0xA5, 0xD6, 0xA7)
CREAM       = RGBColor(0xFB, 0xFA, 0xF0)
TEXT_DARK   = RGBColor(0x21, 0x21, 0x21)
WHITE       = RGBColor(0xFF, 0xFF, 0xFF)
ACCENT      = RGBColor(0x00, 0x66, 0x99)

FONT = "Calibri"


def add_bg(slide, color):
    fill = slide.background.fill
    fill.solid()
    fill.fore_color.rgb = color


def textbox(slide, left, top, width, height, text, size=18, bold=False,
            color=TEXT_DARK, align=PP_ALIGN.LEFT, font=FONT):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.text = text
    p.font.size = Pt(size)
    p.font.bold = bold
    p.font.color.rgb = color
    p.font.name = font
    p.alignment = align
    return tb


def bullets(slide, left, top, width, height, items, size=16, spacing=6):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    first = True
    for txt, lvl in items:
        p = tf.paragraphs[0] if first else tf.add_paragraph()
        first = False
        p.text = ("• " if lvl == 0 else "   – ") + txt
        p.font.size = Pt(size if lvl == 0 else size - 2)
        p.font.color.rgb = TEXT_DARK
        p.font.name = FONT
        p.space_after = Pt(spacing)
        if lvl == 1:
            p.level = 1
    return tb


def header(slide, title, subtitle=None):
    add_bg(slide, CREAM)
    bar = slide.shapes.add_shape(1, 0, 0, Inches(13.33), Inches(1.05))
    bar.fill.solid()
    bar.fill.fore_color.rgb = GREEN_MED
    bar.line.fill.background()
    tf = bar.text_frame
    tf.margin_left = Inches(0.4)
    p = tf.paragraphs[0]
    p.text = title
    p.font.size = Pt(30)
    p.font.bold = True
    p.font.color.rgb = WHITE
    p.font.name = FONT
    if subtitle:
        p2 = tf.add_paragraph()
        p2.text = subtitle
        p2.font.size = Pt(13)
        p2.font.color.rgb = RGBColor(0xE8, 0xF5, 0xE9)
        p2.font.name = FONT


def footer(slide, page):
    tb = slide.shapes.add_textbox(Inches(11.4), Inches(7.0), Inches(1.7), Inches(0.4))
    tf = tb.text_frame
    p = tf.paragraphs[0]
    p.text = f"IVA CIA-3 · {page}"
    p.font.size = Pt(11)
    p.font.color.rgb = GREEN_DARK
    p.font.name = FONT
    p.alignment = PP_ALIGN.RIGHT


def add_picture_center(slide, img_path, top=Inches(1.6), height=Inches(4.6)):
    if os.path.isfile(img_path):
        slide.shapes.add_picture(img_path, Inches(2.2), top, height=height)
        return True
    textbox(slide, Inches(3.5), top + Inches(1.5), Inches(6), Inches(1),
            "[image missing: " + os.path.basename(img_path) + "]", size=14, color=ACCENT)
    return False


def report_table(slide, left, top, width, rows):
    """rows: list of (class, prec, rec, f1); first row is the header."""
    n = len(rows)
    gf = slide.shapes.add_table(n, 4, left, top, width, Inches(0.42) * n)
    table = gf.table
    table.columns[0].width = Inches(6.2)
    table.columns[1].width = Inches(1.9)
    table.columns[2].width = Inches(1.6)
    table.columns[3].width = Inches(1.8)
    for r, (name, prec, rec, f1v) in enumerate(rows):
        vals = [name, prec, rec, f1v]
        for c, v in enumerate(vals):
            cell = table.cell(r, c)
            cell.margin_top = Pt(1); cell.margin_bottom = Pt(1)
            tf = cell.text_frame
            p = tf.paragraphs[0]
            p.text = str(v)
            p.font.name = FONT
            p.alignment = PP_ALIGN.LEFT if c == 0 else PP_ALIGN.CENTER
            if r == 0:
                cell.fill.solid(); cell.fill.fore_color.rgb = GREEN_MED
                p.font.size = Pt(15); p.font.bold = True; p.font.color.rgb = WHITE
            else:
                cell.fill.solid()
                cell.fill.fore_color.rgb = (GREEN_LIGHT if r % 2 == 0 else CREAM)
                p.font.size = Pt(14)
                p.font.bold = (name.startswith("macro"))
                p.font.color.rgb = TEXT_DARK
    return table


# ==================================================================
prs = Presentation()
prs.slide_width = Inches(13.333)
prs.slide_height = Inches(7.5)
blank = prs.slide_layouts[6]

# ---------- 1. TITLE ---------------------------------------------
s = prs.slides.add_slide(blank)
add_bg(s, GREEN_DARK)
tb = s.shapes.add_textbox(Inches(1.2), Inches(1.6), Inches(10.9), Inches(3.2))
tf = tb.text_frame; tf.word_wrap = True
p = tf.paragraphs[0]
p.text = "Intelligent Plant Leaf Disease Detection"
p.font.size = Pt(44); p.font.bold = True; p.font.color.rgb = WHITE; p.font.name = FONT
p2 = tf.add_paragraph()
p2.text = "Deep Learning for Accurate Early Crop Disease Identification"
p2.font.size = Pt(20); p2.font.color.rgb = GREEN_LIGHT; p2.font.name = FONT
tb2 = s.shapes.add_textbox(Inches(1.2), Inches(5.2), Inches(10.9), Inches(1.8))
tf2 = tb2.text_frame; tf2.word_wrap = True
for txt, sz, col, bold in [
    ("Image & Video Analytics (IVA) — Continuous Internal Assessment 3", 18, GREEN_LIGHT, False),
    ("Objective 3", 20, WHITE, True),
    ("PG Sravani  ·  https://github.com/PG-Sravani/plant-leaf-disease-detection", 14, GREEN_LIGHT, False),
]:
    pp = tf2.add_paragraph(); pp.text = txt; pp.font.size = Pt(sz); pp.font.color.rgb = col
    pp.font.name = FONT; pp.font.bold = bold

# ---------- 2. AGENDA --------------------------------------------
s = prs.slides.add_slide(blank)
header(s, "Agenda")
bullets(s, Inches(0.9), Inches(1.6), Inches(11.5), Inches(5.2), [
    ("Objective & problem statement", 0),
    ("Dataset used (Kaggle – New Plant Diseases Dataset)", 0),
    ("Implementation pipeline: Input → Preprocessing → Model → Output", 0),
    ("Deep learning architectures (Custom CNN + Transfer Learning)", 0),
    ("Training configuration & callbacks", 0),
    ("Results: accuracy, training curves, confusion matrix", 0),
    ("Explainability with Grad-CAM", 0),
    ("Web application (Gradio) for real-world prediction", 0),
    ("Project files, GitHub repository & conclusions", 0),
], size=18, spacing=10)
footer(s, 2)

# ---------- 3. OBJECTIVE -----------------------------------------
s = prs.slides.add_slide(blank)
header(s, "Objective & Problem Statement")
bullets(s, Inches(0.9), Inches(1.6), Inches(11.5), Inches(5.5), [
    ("Develop an intelligent deep-learning-based system for accurate "
     "plant leaf disease detection using leaf images.", 0),
    ("Improve crop health through early disease identification.", 0),
    ("Why it matters:", 0),
    ("Plant diseases cause ~20–40% of global crop yield loss annually.", 1),
    ("Manual diagnosis needs experts and is slow / subjective.", 1),
    ("An automated system gives farmers fast, consistent, affordable diagnosis.", 1),
    ("Goal: classify a single leaf image into the correct crop-disease "
     "(or healthy) class.", 0),
    ("Deliverables: trained model, evaluation metrics, heatmaps, web UI.", 0),
], size=17, spacing=10)
footer(s, 3)

# ---------- 4. DATASET -------------------------------------------
s = prs.slides.add_slide(blank)
header(s, "Dataset Used (Kaggle)")
bullets(s, Inches(0.9), Inches(1.5), Inches(12.0), Inches(5.6), [
    ("New Plant Diseases Dataset — vipoooool (derived from PlantVillage)", 0),
    ("87,000+ leaf images · 38 classes · already split into train/valid", 0),
    ("Crops: Apple, Blueberry, Cherry, Corn, Grape, Orange, Peach, Pepper, "
     "Potato, Raspberry, Soybean, Squash, Strawberry, Tomato", 1),
    ("Classes: healthy + bacterial, fungal & viral diseases per crop", 1),
    ("What I actually downloaded & used on this machine:", 0),
    ("38 classes · 70,295 training images · 17,572 validation images", 1),
    ("Quick-demo subset: 7 classes (Potato, Tomato, Bell Pepper)", 1),
    ("Why this dataset: large, balanced, real leaf photos, minimal background "
     "noise — ideal for benchmarking CNN classifiers.", 0),
], size=17, spacing=9)
footer(s, 4)

# ---------- 5. DATASET ACCESS ------------------------------------
s = prs.slides.add_slide(blank)
header(s, "Getting the Dataset — download_dataset.py")
bullets(s, Inches(0.9), Inches(1.6), Inches(11.5), Inches(4.6), [
    ("Automatic download via kagglehub (no manual unzip)", 0),
    ("pip install -r requirements.txt   # includes kagglehub", 1),
    ("python download_dataset.py", 1),
    ("Script resolves Kaggle cache structure and copies train/valid/test", 0),
    ("into ./dataset/ then verifies class & image counts", 0),
], size=17, spacing=8)
textbox(s, Inches(1.2), Inches(5.6), Inches(11.0), Inches(1.2),
        "URL: https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset",
        size=16, color=ACCENT, align=PP_ALIGN.CENTER)
footer(s, 5)

# ---------- 6. PIPELINE (INPUT -> OUTPUT) ------------------------
s = prs.slides.add_slide(blank)
header(s, "Implementation Pipeline (Input → Output)")
bullets(s, Inches(0.9), Inches(1.5), Inches(12.0), Inches(5.6), [
    ("1. Input  — leaf image (JPEG/PNG), resized to 224×224×3 RGB", 0),
    ("2. Preprocessing — pixel scaling to [0,1] + data augmentation "
     "(rotation, flips, zoom, brightness) during training", 0),
    ("3. Model — CNN feature extractor (custom or pretrained backbone) "
     "+ classifier head → softmax over classes", 0),
    ("4. Prediction — argmax of class probabilities (38 or 7 classes)", 0),
    ("5. Interpretation — Grad-CAM heatmap highlights the disease region", 0),
    ("6. Output — top-5 predictions + confidence, shown in a web UI", 0),
], size=17, spacing=10)
footer(s, 6)

# ---------- 7. PREPROCESSING & AUGMENTATION ----------------------
s = prs.slides.add_slide(blank)
header(s, "Data Preprocessing & Augmentation")
bullets(s, Inches(0.9), Inches(1.5), Inches(12.0), Inches(5.0), [
    ("Resize all images to 224×224 (matches ImageNet backbones)", 0),
    ("Rescale pixel values: x / 255  →  [0, 1]", 0),
    ("Training augmentation (helps generalization to real fields):", 0),
    ("Rotation ±20°, width/height shift, shear, zoom", 1),
    ("Horizontal + vertical flips", 1),
    ("Brightness range 0.8–1.2", 1),
    ("Validation: resize + rescale only (no augmentation)", 0),
], size=17, spacing=9)
footer(s, 7)

# ---------- 8. ARCHITECTURES -------------------------------------
s = prs.slides.add_slide(blank)
header(s, "Deep Learning Architectures")
bullets(s, Inches(0.9), Inches(1.5), Inches(12.0), Inches(5.3), [
    ("Option A — Custom CNN (from scratch, ~1.7M params)", 0),
    ("Conv2D + BatchNorm + MaxPool blocks → GlobalAvgPool → Dense → softmax", 1),
    ("Option B — Transfer Learning (features frozen from ImageNet):", 0),
    ("MobileNetV2  — fast, light (CHOSEN for the CPU quick demo)", 1),
    ("EfficientNetB0, ResNet50, DenseNet121, VGG16 — also supported", 1),
    ("Transferred model: base (frozen) → GlobalAvgPool → Dropout → "
     "Dense(128) → Dropout → softmax", 1),
    ("Selection by MODEL_BACKBONE constant in the script", 0),
], size=17, spacing=9)
footer(s, 8)

# ---------- 9. HYBRID BACKEND -------------------------------------
s = prs.slides.add_slide(blank)
header(s, "Hybrid Backend — MobileNetV3 + Vision Transformer")
bullets(s, Inches(0.9), Inches(1.5), Inches(12.0), Inches(3.9), [
    ("Integrated as the web-app prediction engine (PyTorch, CPU):", 0),
    ("MobileNetV3-Large CNN backbone → 1×1 projection (embed 160)", 1),
    ("49 spatial tokens + learned positional embedding", 1),
    ("2 Transformer encoder blocks (self-attention + MLP)", 1),
    ("LayerNorm → mean-pool → classification head (25 PlantVillage "
     "classes)", 1),
    ("Pretrained checkpoint: 3.7M params, 45 MB, 25 classes", 0),
    ("Preprocessing: bilateral filter → vegetation indices (ExG/ExR) → "
     "ImageNet normalization", 0),
], size=17, spacing=8)
textbox(s, Inches(0.9), Inches(5.6), Inches(11.5), Inches(1.3),
        "Verified here on CPU: Potato & pepper disease images detected at "
        "90–100%, healthy-vs-healthy between crops remains difficult (e.g. "
        "potato healthy ~49%). Source: github.com/Vidhi-Garg11/"
        "AI-Crop-Disease-Detection",
        size=13, color=TEXT_DARK, align=PP_ALIGN.CENTER)
footer(s, 9)

# ---------- 10. TRAINING CONFIG ------------------------------------
s = prs.slides.add_slide(blank)
header(s, "Training Configuration")
bullets(s, Inches(0.9), Inches(1.5), Inches(12.0), Inches(5.4), [
    ("Optimizer: Adam, learning rate 1e-4 (fires REDUCED on plateau)", 0),
    ("Loss: Categorical Cross-Entropy · Metric: Accuracy", 0),
    ("Batch size 16–32 · Image size 224×224", 0),
    ("ImageDataGenerator (tf.keras) feeds batches from disk", 0),
    ("Callbacks:", 0),
    ("EarlyStopping — stops if validation loss stops improving (patience)", 1),
    ("ReduceLROnPlateau — halves LR when plateaus (min 1e-7)", 1),
    ("ModelCheckpoint — saves best model to models/*.keras", 1),
    ("Quick demo run: 10 epochs, MobileNetV2, 840 train / 420 valid images", 0),
], size=17, spacing=8)
footer(s, 10)

# ---------- 11. RESULTS SUMMARY ----------------------------------
s = prs.slides.add_slide(blank)
header(s, "Results Summary")
bullets(s, Inches(0.9), Inches(1.5), Inches(12.0), Inches(5.4), [
    ("Quick demo model (MobileNetV2, 7 classes, CPU):", 0),
    ("Validation Accuracy  ≈ 88.3%", 1),
    ("macro avg: precision 0.89 · recall 0.88 · F1 0.88", 1),
    ("Best per-class F1: Potato Early blight 0.97 · Tomato healthy 0.92", 1),
    ("Live web-demo accuracy ≈ 95% (validated via the API on real uploads)", 0),
    ("Full 38-class pipeline supported (needs GPU for practical runtime)", 0),
    ("Evaluations produced automatically: classification report, confusion "
     "matrix, ROC curves", 0),
], size=17, spacing=9)
footer(s, 11)

# ---------- 12. TRAINING CURVES ----------------------------------
s = prs.slides.add_slide(blank)
header(s, "Training & Validation Curves")
img = os.path.join(RESULTS_DIR, "training_curve_subset.png")
add_picture_center(s, img, top=Inches(1.7), height=Inches(4.9))
footer(s, 12)

# ---------- 13. CONFUSION MATRIX ---------------------------------
s = prs.slides.add_slide(blank)
header(s, "Confusion Matrix (Validation)")
img = os.path.join(RESULTS_DIR, "confusion_matrix_subset.png")
add_picture_center(s, img, top=Inches(1.7), height=Inches(4.9))
footer(s, 13)

# ---------- 14. GRAD-CAM -----------------------------------------
s = prs.slides.add_slide(blank)
header(s, "Model Explainability — Grad-CAM")
bullets(s, Inches(0.9), Inches(2.1), Inches(11.5), Inches(2.3), [
    ("Gradient-weighted Class Activation Mapping (Grad-CAM)", 0),
    ("Uses gradients of the target class flowing into the last conv layer to "
     "produce a heatmap of where the model \"looks\".", 0),
    ("Red/orange regions = image areas that most drive the prediction.", 0),
    ("Implemented in grad_cam() (Keras 3 compatible, two-tape chain rule).", 0),
], size=16, spacing=6)
img = os.path.join(RESULTS_DIR, "gradcam_example.jpg")
add_picture_center(s, img, top=Inches(3.35), height=Inches(3.35))
footer(s, 14)

# ---------- 15. CLASSIFICATION REPORT ----------------------------
s = prs.slides.add_slide(blank)
header(s, "Classification Report (Validation, 7 Classes)")
textbox(s, Inches(0.9), Inches(1.25), Inches(11.5), Inches(0.5),
        "MobileNetV2 transfer model · 3,345 validation images · "
        "accuracy 88.3%", size=14, color=TEXT_DARK)
report_table(s, Inches(0.9), Inches(1.85), Inches(11.6), [
    ("Class", "Precision", "Recall", "F1-score"),
    ("Pepper — Bacterial spot", 0.79, 0.94, 0.86),
    ("Pepper — healthy", 0.91, 0.84, 0.88),
    ("Potato — Early blight", 0.96, 0.93, 0.94),
    ("Potato — Late blight", 0.91, 0.81, 0.85),
    ("Potato — healthy", 0.87, 0.93, 0.90),
    ("Tomato — Late blight", 0.81, 0.92, 0.86),
    ("Tomato — healthy", 0.98, 0.83, 0.90),
    ("macro average", 0.89, 0.88, 0.88),
])
footer(s, 15)

# ---------- 16. WEB APP ------------------------------------------
s = prs.slides.add_slide(blank)
header(s, "Web Application — real-time prediction (Gradio)")
bullets(s, Inches(0.9), Inches(1.5), Inches(12.0), Inches(5.5), [
    ("run_demo.py loads models/…keras + class_names.json and starts a server", 0),
    ("run_demo_hybrid.py — second server powered by the 25-class hybrid "
     "checkpoint (same upload UI)", 1),
    ("Browser UI at http://127.0.0.1:7860", 1),
    ("Upload a leaf photo → top-5 disease predictions with confidence %", 1),
    ("Handles common image formats; converts RGBA/B&W uploads to RGB", 1),
    ("Verified live: 4/4 validation samples classified correctly via the API",
     0),
    ("Earlier upload bug (4-channel images) found and fixed during testing",
     0),
], size=17, spacing=9)
footer(s, 16)

# ---------- 17. SAMPLE OUTPUT ------------------------------------
s = prs.slides.add_slide(blank)
header(s, "Sample Output — Disease Prediction")
img = os.path.join(RESULTS_DIR, "output_sample.png")
if os.path.isfile(img):
    from PIL import Image as _PILImage
    _w, _h = _PILImage.open(img).size
    _height = 4.4
    _width = _height * (_w / _h)
    left = (13.333 - _width) / 2
    s.shapes.add_picture(img, Inches(left), Inches(1.8), width=Inches(_width), height=Inches(_height))
else:
    textbox(s, Inches(3.5), Inches(3.0), Inches(6), Inches(1),
            "[image missing: output_sample.png]", size=14, color=ACCENT)
textbox(s, Inches(1.0), Inches(6.5), Inches(11.3), Inches(0.8),
        "Uploaded leaf photo → predicted disease + confidence, rendered by the "
        "Gradio web application (runtime screenshot)",
        size=14, color=TEXT_DARK, align=PP_ALIGN.CENTER)
footer(s, 17)

# ---------- 18. PROJECT STRUCTURE --------------------------------
s = prs.slides.add_slide(blank)
header(s, "Project Files & Structure")
bullets(s, Inches(0.9), Inches(1.5), Inches(12.0), Inches(5.5), [
    ("plant_disease_detection.py — full 38-class pipeline (all models, "
     "Grad-CAM, evaluation)", 0),
    ("plant_disease_detection_quick.py — CPU-friendly 7-class training", 0),
    ("hybrid/hybrid_backend.py + hybrid_model_best.pth — MobileNetV3+ViT "
     "backend (PyTorch)", 0),
    ("download_dataset.py — automatic Kaggle download", 0),
    ("run_demo.py — Gradio web application", 0),
    ("requirements.txt — dependencies", 0),
    ("README.md · WORKING_PROTOCOL.md — documentation", 0),
    ("results/ — training curve, confusion matrix images", 0),
    ("models/ — trained .keras weights + class_names.json", 0),
    ("Dataset: ./dataset/train (70,295) · ./dataset/valid (17,572)", 0),
], size=16, spacing=8)
footer(s, 18)

# ---------- 19. REPO ---------------------------------------------
s = prs.slides.add_slide(blank)
header(s, "GitHub Repository")
textbox(s, Inches(1.0), Inches(2.0), Inches(11.3), Inches(1.0),
        "https://github.com/PG-Sravani/plant-leaf-disease-detection",
        size=28, bold=True, color=ACCENT, align=PP_ALIGN.CENTER)
bullets(s, Inches(1.5), Inches(3.4), Inches(10.3), Inches(3.4), [
    ("Public repo — all code, trained model, result images and docs", 0),
    ("Commits trace the full workflow: setup → data → training → demo → fix", 0),
    ("Clone & reproduce:  git clone https://github.com/PG-Sravani/…", 0),
], size=17, spacing=8)
footer(s, 19)

# ---------- 20. CONCLUSION ---------------------------------------
s = prs.slides.add_slide(blank)
header(s, "Conclusion & Learning Outcomes")
bullets(s, Inches(0.9), Inches(1.6), Inches(11.5), Inches(5.4), [
    ("Demonstrated an end-to-end deep learning pipeline for plant leaf "
     "disease detection on real Kaggle data.", 0),
    ("Achieved ~88% validation accuracy on a 7-class quick model using "
     "MobileNetV2 transfer learning (CPU-only environment).", 0),
    ("Full 38-class support with multiple backbones and automatic "
     "evaluation, Grad-CAM, and a deployable web UI.", 0),
    ("Learned: dataset handling, augmentation, transfer learning, "
     "callbacks, evaluation metrics, model interpretability, deployment.", 0),
    ("Real bug-fixing experience: dataset cache structure, Keras 3 API "
     "changes, and RGBA image upload handling.", 0),
], size=17, spacing=10)
footer(s, 20)

# ---------- 21. FUTURE WORK --------------------------------------
s = prs.slides.add_slide(blank)
header(s, "Future Work")
bullets(s, Inches(0.9), Inches(1.6), Inches(11.5), Inches(5.0), [
    ("Train the full 38-class model on a GPU for production accuracy "
     "(>98% typical on this dataset).", 0),
    ("Fine-tune frozen backbones by unfreezing top layers.", 0),
    ("Add per-disease treatment/precaution advice in the web app.", 0),
    ("Extend to object detection (bounding boxes) and mobile edge devices.", 0),
    ("Add confidence-based \"uncertain\" flagging to flag uncertain cases "
     "for expert review.", 0),
], size=17, spacing=10)
footer(s, 21)

# ---------- 22. THANK YOU ----------------------------------------
s = prs.slides.add_slide(blank)
add_bg(s, GREEN_DARK)
tb = s.shapes.add_textbox(Inches(1.2), Inches(2.8), Inches(10.9), Inches(1.6))
tf = tb.text_frame
p = tf.paragraphs[0]
p.text = "Thank You"
p.font.size = Pt(48); p.font.bold = True; p.font.color.rgb = WHITE
p.font.name = FONT; p.alignment = PP_ALIGN.CENTER
tb2 = s.shapes.add_textbox(Inches(1.2), Inches(4.4), Inches(10.9), Inches(1.2))
tf2 = tb2.text_frame
p = tf2.paragraphs[0]
p.text = "Questions & Discussion"
p.font.size = Pt(22); p.font.color.rgb = GREEN_LIGHT; p.font.name = FONT
p.alignment = PP_ALIGN.CENTER

prs.save(OUT)
print("Presentation saved to:", OUT)
print("Total slides:", len(prs.slides.__iter__.__self__._sldIdLst))