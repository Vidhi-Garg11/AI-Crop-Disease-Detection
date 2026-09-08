"""
=====================================================================
 Intelligent Plant Leaf Disease Detection System
 Objective 3 - Image and Video Analytics (IVA CIA-3)
=====================================================================

DEVELOPER NOTE: This script trains a deep learning CNN to classify
plant leaf diseases using leaf images. It uses the famous Kaggle
"New Plant Diseases Dataset" (PlantVillage-derived, 38 classes).

DATASET REQUIREMENTS (Download from Kaggle):
  Name   : New Plant Diseases Dataset
  Author : vipoooool
  URL    : https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

  Extract the 'New Plant Diseases Dataset (Augmented)' folder and point
  DATASET_PATH below to the inner 'New Plant Diseases Dataset (Augmented)'
  directory which contains 'train' and 'valid' subfolders.

Folder structure expected:
  New Plant Diseases Dataset (Augmented)/
    train/
      Apple___Apple_scab/
      Apple___Black_rot/
      ... (38 classes)
    valid/
      Apple___Apple_scab/
      ...

Optional Test set (separate Kaggle download):
  Name   : Plant Disease Detection - Test Set Images (for New Plant Diseases)
  This gives 'test' folder with indexed images used for the final model.

This script includes:
  1. Custom CNN (from scratch, no pretraining)
  2. Transfer Learning with VGG16 / EfficientNetB0 / ResNet50 / MobileNetV2
  3. Gradient-weighted Class Activation Mapping (Grad-CAM) visualization
  4. Web-based inference (Gradio) for real-world usage
  5. Full metrics, confusion matrix, ROC curves
  6. Model persistence (.h5 / .keras)
=====================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless safe
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import warnings
warnings.filterwarnings("ignore")

# ----------------------------------------------------------------
# TensorFlow / Keras
# ----------------------------------------------------------------
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import (VGG16, EfficientNetB0, ResNet50,
                                           MobileNetV2, DenseNet121)
from tensorflow.keras.applications.vgg16 import preprocess_input as vgg_pre
from tensorflow.keras.applications.efficientnet import preprocess_input as eff_pre
from tensorflow.keras.applications.resnet50 import preprocess_input as res_pre
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input as mob_pre
from tensorflow.keras.applications.densenet import preprocess_input as dns_pre
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam

from sklearn.metrics import (classification_report, confusion_matrix,
                             accuracy_score, roc_auc_score, roc_curve)
from sklearn.preprocessing import LabelEncoder

# ----------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------
# CHANGE THIS PATH to your extracted dataset location
DATASET_PATH = r"C:\Users\PG Sravani\Downloads\IVA CIA-3\dataset"
TRAIN_DIR    = os.path.join(DATASET_PATH, "train")
VALID_DIR    = os.path.join(DATASET_PATH, "valid")
TEST_DIR     = os.path.join(DATASET_PATH, "test")   # optional, may not exist

IMG_SIZE     = 224
BATCH_SIZE   = 32
EPOCHS       = 15
LEARNING_RATE = 1e-4

RESULTS_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MODELS_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

MODEL_BACKBONE = "custom"   # options: custom | VGG16 | EfficientNetB0 | ResNet50 | MobileNetV2 | DenseNet121


# ----------------------------------------------------------------
# 1. DATA PREPARATION
# ----------------------------------------------------------------
def show_diagnostics():
    print("=" * 70)
    print("TENSORFLOW / HARDWARE DIAGNOSTICS")
    print("=" * 70)
    print("TensorFlow version :", tf.__version__)
    print("Keras version      :", keras.__version__)
    gpus = tf.config.list_physical_devices("GPU")
    print("GPUs available     :", gpus if gpus else "None (using CPU)")
    if gpus:
        for g in gpus:
            print("   ->", g)
    print()


def build_generators():
    """Create training / validation data generators with augmentation."""
    train_aug = ImageDataGenerator(
        rescale=1.0 / 255.0,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode="nearest",
    )

    valid_gen = ImageDataGenerator(rescale=1.0 / 255.0)

    train_flow = train_aug.flow_from_directory(
        TRAIN_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=True)

    valid_flow = valid_gen.flow_from_directory(
        VALID_DIR, target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, class_mode="categorical", shuffle=False)

    return train_flow, valid_flow


# ----------------------------------------------------------------
# 2. MODEL ARCHITECTURES
# ----------------------------------------------------------------
def build_custom_cnn(num_classes):
    """A compact CNN built from scratch (no transfer learning)."""
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(256, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D(2, 2),
        layers.Conv2D(512, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.5),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def build_transfer_model(backbone, num_classes, img_size=IMG_SIZE):
    """Build a transfer-learning model with a frozen pretrained backbone."""
    pre_map = {
        "VGG16": (VGG16, vgg_pre),
        "EfficientNetB0": (EfficientNetB0, eff_pre),
        "ResNet50": (ResNet50, res_pre),
        "MobileNetV2": (MobileNetV2, mob_pre),
        "DenseNet121": (DenseNet121, dns_pre),
    }
    if backbone not in pre_map:
        raise ValueError(f"Backbone {backbone} not supported")

    base_cls, _ = pre_map[backbone]
    base = base_cls(weights="imagenet", include_top=False,
                    input_shape=(img_size, img_size, 3))
    base.trainable = False   # freeze

    model = models.Sequential([
        base,
        layers.GlobalAveragePooling2D(),
        layers.Dropout(0.3),
        layers.Dense(512, activation="relu"),
        layers.BatchNormalization(),
        layers.Dropout(0.4),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


# ----------------------------------------------------------------
# 3. TRAINING LOOP WITH CALLBACKS
# ----------------------------------------------------------------
def train_model(train_flow, valid_flow, num_classes, backbone):
    if backbone == "custom":
        model = build_custom_cnn(num_classes)
        print(f"[INFO] Training custom CNN with {num_classes} classes")
    else:
        model = build_transfer_model(backbone, num_classes)
        print(f"[INFO] Training transfer-learning model with {backbone} backbone")

    model.compile(optimizer=Adam(learning_rate=LEARNING_RATE),
                  loss="categorical_crossentropy",
                  metrics=["accuracy"])
    model.summary()

    callbacks = [
        EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True,
                      verbose=1),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                          min_lr=1e-7, verbose=1),
        ModelCheckpoint(os.path.join(MODELS_DIR, f"best_{backbone}.keras"),
                        monitor="val_accuracy", save_best_only=True, verbose=1),
    ]

    start = time.time()
    history = model.fit(
        train_flow,
        validation_data=valid_flow,
        epochs=EPOCHS,
        callbacks=callbacks,
        verbose=1,
    )
    elapsed = (time.time() - start) / 60.0
    print(f"\n[INFO] Training completed in {elapsed:.2f} minutes")

    # Save final model
    model_path = os.path.join(MODELS_DIR, f"plant_disease_{backbone}_final.keras")
    model.save(model_path)
    print(f"[INFO] Model saved to: {model_path}")

    return model, history


# ----------------------------------------------------------------
# 4. PLOTTING / EVALUATION HELPERS
# ----------------------------------------------------------------
def plot_history(history, backbone):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, metric, ylabel in zip(axes, ["accuracy", "loss"],
                                  ["Accuracy", "Loss"]):
        ax.plot(history.history[metric], label=f"Train {ylabel}")
        ax.plot(history.history[f"val_{metric}"], label=f"Valid {ylabel}")
        ax.set_title(f"{backbone} - {ylabel} over Epochs")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.legend()
        ax.grid(True)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"training_curve_{backbone}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Training curves saved -> {path}")


def evaluate_model(model, valid_flow, class_names, backbone):
    """Evaluate and produce: classification report, confusion matrix, ROC."""
    nb_batches = int(np.ceil(valid_flow.samples / valid_flow.batch_size))
    Y_pred = model.predict(valid_flow, steps=nb_batches, verbose=1)
    y_pred = np.argmax(Y_pred, axis=1)

    # Rebuild true labels in class order (generator is shuffled=False)
    y_true = valid_flow.classes[: len(y_pred)]
    labels = list(class_names)

    acc = accuracy_score(y_true, y_pred)
    print("\n" + "=" * 70)
    print(f"FINAL VALIDATION RESULTS  ({backbone})")
    print("=" * 70)
    print(f"Validation Accuracy : {acc:.4f}")

    report = classification_report(y_true, y_pred, target_names=labels,
                                   zero_division=0)
    print("\nClassification Report:\n", report)

    # ------- Confusion Matrix -------
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(20, 18))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels,
                yticklabels=labels)
    plt.title(f"Confusion Matrix - {backbone}")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"confusion_matrix_{backbone}.png")
    plt.savefig(path, dpi=140)
    plt.close()
    print(f"[INFO] Confusion matrix saved -> {path}")

    # ------- ROC Curves (One-vs-Rest) -------
    try:
        # limit to a subset of classes for readability
        n_classes = len(labels)
        top = min(n_classes, 10)
        plt.figure(figsize=(12, 10))
        for i in range(top):
            fpr, tpr, _ = roc_curve((y_true == i).astype(int), Y_pred[:, i])
            auc = roc_auc_score((y_true == i).astype(int), Y_pred[:, i])
            plt.plot(fpr, tpr, label=f"{labels[i]} (AUC={auc:.3f})")
        plt.plot([0, 1], [0, 1], "k--", label="Random")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"ROC Curves (One-vs-Rest) - {backbone} (top {top} classes)")
        plt.legend(loc="best", fontsize=8)
        plt.grid(True)
        plt.tight_layout()
        path = os.path.join(RESULTS_DIR, f"roc_curve_{backbone}.png")
        plt.savefig(path, dpi=140)
        plt.close()
        print(f"[INFO] ROC curves saved -> {path}")
    except Exception as e:
        print("[WARN] ROC computation failed:", e)

    return acc, report


# ----------------------------------------------------------------
# 5. GRAD-CAM VISUALIZATION
# ----------------------------------------------------------------
def grad_cam(model, img_array, class_idx, layer_name=None):
    """Produce a Grad-CAM heatmap for a transfer-learning model (Keras 3 compatible).

    Keras 3.15 exposes layer .output only after a call and cannot reach layers
    nested inside a base Functional via model.get_layer. We therefore rebuild
    the forward graph manually: a bridge model on the base input produces both
    the last-conv activations and the base features, a fresh classifier copies
    the tail weights, and a two-tape chain-rule computes dP/d(conv) exactly.
    """
    # find the last conv layer, and the enclosing base sub-model
    last_conv, base_parent = None, None
    for layer in reversed(model.layers):
        if isinstance(layer, layers.Conv2D):
            last_conv, base_parent = layer, model
            break
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, layers.Conv2D):
                    last_conv, base_parent = sub, layer
                    break
        if last_conv:
            break
    if layer_name is not None:
        for layer in model.layers:
            chain = layer.layers if hasattr(layer, "layers") else [layer]
            for cand in chain:
                if cand.name == layer_name and isinstance(cand, layers.Conv2D):
                    last_conv, base_parent = cand, layer
        if last_conv is None:
            raise ValueError("Layer %r not found / not a conv layer" % layer_name)
    if last_conv is None:
        raise ValueError("No Conv layer found for Grad-CAM")

    base = base_parent if base_parent is not None else model
    _ = model(np.ones((1, 224, 224, 3), dtype="float32"))  # warm call

    # 1) bridge model: conv activations + base feature map (same graph)
    gmod = models.Model([base.input], [last_conv.output, base.output])

    # 2) fresh classifier on top of the base features with copied weights
    tail = [l for l in model.layers if l is not base]
    inp = layers.Input(shape=tuple(base.output.shape[1:]))
    x = inp
    for l in tail:
        x = l(x) if not isinstance(l, layers.Dropout) else layers.Dropout(l.rate)(x)
    clf = models.Model(inp, x)
    for new, old in zip(clf.layers[1:], tail):
        new.set_weights(old.get_weights())

    img = tf.convert_to_tensor(np.expand_dims(img_array, 0), dtype="float32")
    img = tf.Variable(img)
    gB = None
    for _ in range(4):
        with tf.GradientTape() as tape1:
            A, B = gmod(img)
            logits = clf(B)
            gB = tape1.gradient(logits[0, class_idx], B)
        if gB is not None:
            break
    if gB is None:
        raise ValueError("gradient to base features is None")
    with tf.GradientTape() as tape2:
        A2, B2 = gmod(img)
        loss = tf.reduce_sum(B2 * gB)
    gA = tape2.gradient(loss, A2)
    if gA is None:
        raise ValueError("gradient to conv activations is None")

    weights = tf.reduce_mean(gA, axis=(0, 1, 2))
    heatmap = tf.reduce_sum(A2[0] * weights[None, None, :], axis=-1)
    heatmap = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-10)
    return heatmap.numpy(), last_conv.name


def visualize_gradcam(model, flow, class_names, backbone, num_samples=5):
    """Overlay Grad-CAM on sample validation images."""
    samples = []
    batch_imgs, batch_labels = next(flow)
    nb_batches = int(np.ceil(flow.samples / flow.batch_size))
    for _ in range(min(nb_batches, 10)):
        bi, bl = next(flow)
        samples.append((bi, bl))
        if len(samples) >= 3:
            break

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 4))
    if num_samples == 1:
        axes = axes[np.newaxis, :]
    count = 0
    for bi, bl in samples:
        for i in range(len(bi)):
            if count >= num_samples:
                break
            img = bi[i]
            lbl = int(np.argmax(bl[i]))
            pred_probs = model.predict(np.expand_dims(img, 0), verbose=0)[0]
            pred = int(np.argmax(pred_probs))
            heatmap, layer = grad_cam(model, img, lbl)

            # original
            axes[count, 0].imshow(img)
            axes[count, 0].set_title(f"True: {class_names[lbl]}", fontsize=9)
            axes[count, 0].axis("off")

            # heatmap only
            axes[count, 1].imshow(heatmap, cmap="jet", alpha=0.7)
            axes[count, 1].set_title(f"Grad-CAM (layer: {layer})", fontsize=8)
            axes[count, 1].axis("off")

            # overlay
            heatmap_resized = tf.image.resize(
                heatmap[..., np.newaxis], (IMG_SIZE, IMG_SIZE)).numpy()[..., 0]
            axes[count, 2].imshow(img)
            axes[count, 2].imshow(heatmap_resized, cmap="jet", alpha=0.45)
            axes[count, 2].set_title(
                f"Pred: {class_names[pred]} ({pred_probs[pred]:.2f})", fontsize=9)
            axes[count, 2].axis("off")

            count += 1
    plt.suptitle(f"Grad-CAM Interpretability - {backbone}", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, f"gradcam_{backbone}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[INFO] Grad-CAM visualizations saved -> {path}")


# ----------------------------------------------------------------
# 6. GRADIO WEB INTERFACE (REAL-WORLD PREDICTION)
# ----------------------------------------------------------------
def build_gradio_app(model, class_names):
    """Interactive web UI for uploading a leaf image and predicting disease."""
    try:
        import gradio as gr
    except ImportError:
        print("[WARN] gradio not installed. Run: pip install gradio")
        return

    from tensorflow.keras.preprocessing import image as kimage

    def predict(img):
        img = kimage.img_to_array(img)
        img = np.expand_dims(img, 0) / 255.0
        probs = model.predict(img, verbose=0)[0]
        top_idx = np.argsort(probs)[::-1]
        result = {class_names[i]: float(probs[i]) for i in top_idx[:5]}
        return result

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload a plant leaf image"),
        outputs=gr.Label(num_top_classes=5, label="Predicted Disease"),
        title="🌿 Intelligent Plant Leaf Disease Detection",
        description=(
            "Upload a clear leaf image. The deep learning model will identify "
            "the crop and disease condition (or healthy). Built for IVA CIA-3."
        ),
    )
    demo.launch(share=False)
    print("[INFO] Gradio app launched. Open the URL in your browser.")


# ----------------------------------------------------------------
# MAIN DRIVER
# ----------------------------------------------------------------
def main():
    show_diagnostics()

    if not os.path.isdir(TRAIN_DIR) or not os.path.isdir(VALID_DIR):
        print(f"""
[ERROR] Dataset folders not found at:
  Train: {TRAIN_DIR}
  Valid: {VALID_DIR}

Please download the 'New Plant Diseases Dataset' from Kaggle:
  https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

Extract it, locate the inner folder 'New Plant Diseases Dataset (Augmented)'
(which contains 'train' and 'valid' subfolders), and update DATASET_PATH
at the top of this script to point to it.
""")
        return

    print("[INFO] Building data generators...")
    train_flow, valid_flow = build_generators()

    num_classes = len(train_flow.class_indices)
    class_names = [""] * num_classes
    for name, idx in train_flow.class_indices.items():
        class_names[idx] = name
    print(f"\n[INFO] Found {num_classes} classes:")
    for i, n in enumerate(class_names):
        print(f"   {i:2d}. {n}")
    print()

    # Class distribution in training set
    dist = Counter(train_flow.classes)
    print("[INFO] Training samples per class:")
    for i in range(num_classes):
        print(f"   {class_names[i]:45s} : {dist.get(i, 0)} images")

    model, history = train_model(train_flow, valid_flow, num_classes, MODEL_BACKBONE)

    plot_history(history, MODEL_BACKBONE)

    # Reset validation generator for consistent shuffling (shuffle=False)
    valid_gen = ImageDataGenerator(rescale=1.0 / 255.0)
    valid_flow = valid_gen.flow_from_directory(
        VALID_DIR, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode="categorical", shuffle=False)

    evaluate_model(model, valid_flow, class_names, MODEL_BACKBONE)

    print("\n[INFO] Generating Grad-CAM visualizations...")
    visualize_gradcam(model, valid_flow, class_names, MODEL_BACKBONE)

    print("\n" + "=" * 70)
    print("All outputs are saved in: ")
    print(f"   Results : {RESULTS_DIR}")
    print(f"   Models  : {MODELS_DIR}")
    print("=" * 70)

    # Optionally offer to launch the web app
    answer = input("\nLaunch Gradio web app for real-time prediction? [y/N]: ").strip()
    if answer.lower() in ("y", "yes"):
        build_gradio_app(model, class_names)


if __name__ == "__main__":
    main()
