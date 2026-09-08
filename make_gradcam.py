"""Generate real Grad-CAM heatmap overlays for the quick MobileNetV2 model
using a Keras 3.15-compatible two-tape chain-rule approach.

Outputs:
  results/gradcam_example.jpg   - the primary heatmap-overlay figure
"""
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras import models, layers, Input

OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)


def load_model_parts(model_path):
    m = models.load_model(model_path)
    base = m.get_layer("mobilenetv2_1.00_224")
    conv = [l for l in reversed(base.layers) if isinstance(l, layers.Conv2D)][0]
    _ = m(np.ones((1, 224, 224, 3)))

    gmod = models.Model([base.input], [conv.output, base.output])

    inp = Input(shape=tuple(base.output.shape[1:]))
    x = layers.GlobalAveragePooling2D()(inp)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    x = layers.Dense(7, activation="softmax")(x)
    clf = models.Model(inp, x)
    for new, old in zip(clf.layers[1:], list(m.layers[1:])):
        new.set_weights(old.get_weights())
    return m, gmod, clf


def grad_cam(gmod, clf, img_pre, class_idx):
    """Return (heatmap (7,7) float, A2, logits)."""
    img = tf.Variable(img_pre, dtype=tf.float32)
    gB = None
    for _ in range(4):
        with tf.GradientTape() as tape1:
            A, B = gmod(img)
            logits = clf(B)
            gB = tape1.gradient(logits[0, class_idx], B)
        if gB is not None:
            break
    if gB is None:
        raise RuntimeError("gradient to base output is None")
    with tf.GradientTape() as tape2:
        A2, B2 = gmod(img)
        loss = tf.reduce_sum(B2 * gB)
    gA = tape2.gradient(loss, A2)
    if gA is None:
        raise RuntimeError("gradient to conv activations is None")
    weights = tf.reduce_mean(gA, axis=(0, 1, 2))                   # (1280,)
    heat = tf.reduce_sum(A2[0] * weights[None, None, :], axis=-1)  # (7,7)
    heat = tf.maximum(heat, 0)
    denom = tf.reduce_max(heat)
    heat = heat / (denom + 1e-10)
    return heat.numpy(), A2, logits


def overlay_heatmap(img_rgb, heat, alpha=0.5):
    """Overlay Grad-CAM heatmap (7,7) on RGB image (224,224,3) [0,1]."""
    heat_up = np.array(Image.fromarray((heat * 255).astype(np.uint8)).resize(
        (img_rgb.shape[1], img_rgb.shape[0]), Image.BICUBIC)) / 255.0
    colored = plt.cm.jet(heat_up)[..., :3]
    return (1 - alpha) * img_rgb + alpha * colored


def main():
    print("Loading model parts ...")
    m, gmod, clf = load_model_parts(r"models\plant_disease_subset_final.keras")
    class_names = [c for c in m.layers[-1].config.get("class_names", [])] if False else None
    import json
    with open(r"models\class_names.json") as f:
        class_names = json.load(f)
    print("classes:", class_names)

    sample_specs = [
        (r"dataset\valid\Potato___Early_blight", "Potato Early blight"),
        (r"dataset\valid\Potato___Late_blight", "Potato Late blight"),
        (r"dataset\valid\Tomato___Late_blight", "Tomato Late blight"),
        (r"dataset\valid\Pepper,_bell___Bacterial_spot", "Bell-pepper bacterial spot"),
    ]
    chosen = None
    for folder, _label in sample_specs:
        if os.path.isdir(folder):
            files = sorted(os.listdir(folder))
            if files:
                chosen = os.path.join(folder, files[0])
                break
    if chosen is None:
        raise SystemExit("no sample image found")
    print("sample image:", chosen)

    rgb = np.array(Image.open(chosen).convert("RGB").resize((224, 224)))
    img_pre = rgb.astype(np.float32) / 255.0
    heat, _A2, logits = grad_cam(gmod, clf, img_pre[None], 2)

    pred_idx = int(np.argmax(logits.numpy()[0]))
    probs = logits.numpy()[0]
    print("logits:", np.round(probs, 3).tolist(), "-> pred:", class_names[pred_idx])

    overlay = overlay_heatmap(img_pre, heat)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    axes[0].imshow(rgb)
    axes[0].set_title("Input Leaf", fontsize=13)
    axes[0].axis("off")
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Heatmap Overlay", fontsize=13)
    axes[1].axis("off")
    fig.suptitle(
        "Grad-CAM: '{}'  (conf {:.1%})".format(
            class_names[pred_idx], probs[pred_idx]),
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    out = os.path.join(OUT_DIR, "gradcam_example.jpg")
    fig.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("saved:", out)


if __name__ == "__main__":
    main()