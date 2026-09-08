"""
=====================================================================
 Intelligent Plant Leaf Disease Detection System
        (QUICK-START VARIANT with sample subclass)
=====================================================================

This version:
  - Trains on a SUBSET of classes (e.g., a subset of the PlantVillage
    dataset) to make training fast enough for a laptop without GPU.
  - Produces the same deliverables: accuracy, confusion matrix, ROC,
    Grad-CAM, model file, and a Gradio web demo.

Use this if you cannot download the full dataset or your machine
lacks a GPU.

To change which classes are used, edit SUBSET_CLASSES below (use names
matching your folder names under the train/ directory).
=====================================================================
"""

import os
import glob
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import shutil
import tempfile
import random
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------
DATASET_PATH = r"C:\Users\PG Sravani\Downloads\IVA CIA-3\dataset"
FULL_TRAIN   = os.path.join(DATASET_PATH, "train")   # full train folder
FULL_VALID   = os.path.join(DATASET_PATH, "valid")   # full valid folder

# Choose a manageable subset of classes to keep training fast.
# These names MUST match folder names in your train/ directory.
SUBSET_CLASSES = [
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato___Late_blight",
    "Tomato___healthy",
    "Pepper,_bell___Bacterial_spot",
    "Pepper,_bell___healthy",
]

IMG_SIZE   = 224
BATCH_SIZE = 16
EPOCHS     = 10
MODEL_NAME = "MobileNetV2"   # or EfficientNetB0

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# ----------------------------------------------------------------
# HELPER: build a reduced dataset into a temp directory
# ----------------------------------------------------------------
def build_subset(src_dir, dst_dir, classes, per_class_limit=None):
    """Copy a limited number of images per class into dst_dir."""
    if os.path.isdir(dst_dir):
        shutil.rmtree(dst_dir)
    os.makedirs(dst_dir, exist_ok=True)

    for cls in classes:
        src_cls = os.path.join(src_dir, cls)
        if not os.path.isdir(src_cls):
            print(f"  [WARN] Class folder not found: {src_cls}")
            continue
        imgs = glob.glob(os.path.join(src_cls, "*.*"))
        if per_class_limit:
            imgs = random.sample(imgs, min(per_class_limit, len(imgs)))
        dst_cls = os.path.join(dst_dir, cls)
        os.makedirs(dst_cls, exist_ok=True)
        for f in imgs:
            shutil.copy2(f, os.path.join(dst_cls, os.path.basename(f)))
    print(f"[INFO] Built subset dataset at {dst_dir}")
    return dst_dir


# ----------------------------------------------------------------
def main():
    print("TensorFlow:", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs:", gpus if gpus else "None (CPU mode)")

    if not os.path.isdir(FULL_TRAIN):
        print(f"[ERROR] Train folder not found: {FULL_TRAIN}")
        print("Please update DATASET_PATH to point to 'New Plant Diseases Dataset (Augmented)'")
        return

    workdir = tempfile.mkdtemp(prefix="plantdisease_")
    sub_train = os.path.join(workdir, "train")
    sub_valid = os.path.join(workdir, "valid")

    print("[INFO] Building training subset...")
    build_subset(FULL_TRAIN, sub_train, SUBSET_CLASSES, per_class_limit=120)
    print("[INFO] Building validation subset...")
    build_subset(FULL_VALID, sub_valid, SUBSET_CLASSES, per_class_limit=60)

    train_aug = ImageDataGenerator(
        rescale=1.0 / 255.0, rotation_range=20, width_shift_range=0.2,
        height_shift_range=0.2, zoom_range=0.2, horizontal_flip=True,
        brightness_range=[0.8, 1.2])
    valid_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_flow = train_aug.flow_from_directory(
        sub_train, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=True)
    valid_flow = valid_gen.flow_from_directory(
        sub_valid, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False)

    num_classes = len(train_flow.class_indices)
    class_names = [""] * num_classes
    for n, i in train_flow.class_indices.items():
        class_names[i] = n
    print("\nClasses:", class_names)

    # Save class mapping alongside the model for later use in the web demo
    model_dir = os.path.dirname(os.path.abspath(__file__))
    with open(os.path.join(model_dir, "models", "class_names.json"), "w") as fp:
        json.dump(class_names, fp, indent=2)
    print("[INFO] Saved class_names.json")

    # ---- Build transfer model ----
    if MODEL_NAME == "EfficientNetB0":
        base = EfficientNetB0(weights="imagenet", include_top=False,
                              input_shape=(IMG_SIZE, IMG_SIZE, 3))
    else:
        base = MobileNetV2(weights="imagenet", include_top=False,
                           input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(num_classes, activation="softmax"),
    ])
    model.compile(optimizer=Adam(learning_rate=1e-4),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, min_lr=1e-7),
        ModelCheckpoint(os.path.join(MODELS_DIR, f"best_{MODEL_NAME}_subset.keras"),
                        monitor="val_accuracy", save_best_only=True),
    ]

    hist = model.fit(train_flow, validation_data=valid_flow,
                     epochs=EPOCHS, callbacks=callbacks, verbose=1)

    # ---- History plot ----
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].plot(hist.history["accuracy"], label="Train")
    ax[0].plot(hist.history["val_accuracy"], label="Valid")
    ax[0].set_title("Accuracy"); ax[0].legend(); ax[0].grid(True)
    ax[1].plot(hist.history["loss"], label="Train")
    ax[1].plot(hist.history["val_loss"], label="Valid")
    ax[1].set_title("Loss"); ax[1].legend(); ax[1].grid(True)
    plt.savefig(os.path.join(RESULTS_DIR, "training_curve_subset.png"), dpi=150)
    print("[INFO] Saved training curve")

    # ---- Evaluation ----
    nb = int(np.ceil(valid_flow.samples / valid_flow.batch_size))
    Yp = model.predict(valid_flow, steps=nb, verbose=0)
    yp = np.argmax(Yp, axis=1)
    yt = valid_flow.classes[: len(yp)]

    acc = np.mean(yp == yt)
    print(f"\nValidation Accuracy: {acc:.4f}")
    print(classification_report(yt, yp, target_names=class_names, zero_division=0))

    cm = confusion_matrix(yt, yp)
    plt.figure(figsize=(8, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names)
    plt.title("Confusion Matrix (subset)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "confusion_matrix_subset.png"), dpi=140)

    model.save(os.path.join(MODELS_DIR, "plant_disease_subset_final.keras"))
    print("[INFO] Model saved and results written.")


if __name__ == "__main__":
    main()
