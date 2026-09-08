"""
=====================================================================
 Launch the Gradio Web Demo for Plant Leaf Disease Detection
=====================================================================
Loads a trained model (default: models/plant_disease_subset_final.keras
from the quick training) and serves a browser interface.

Usage:
    python run_demo.py
=====================================================================
"""

import os
import glob
import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import models as keras_models
from tensorflow.keras.preprocessing import image as kimage

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(PROJECT_DIR, "models",
                           "plant_disease_subset_final.keras")
CLASS_MAP   = os.path.join(PROJECT_DIR, "models", "class_names.json")


def get_class_names():
    """Load class names saved during training; fallback to train folder."""
    if os.path.isfile(CLASS_MAP):
        with open(CLASS_MAP, "r") as fp:
            names = json.load(fp)
        return names
    train_dir = os.path.join(PROJECT_DIR, "dataset", "train")
    if os.path.isdir(train_dir):
        return sorted(os.listdir(train_dir))
    return None


def main():
    if not os.path.isfile(MODEL_PATH):
        print("[ERROR] Trained model not found:", MODEL_PATH)
        print("        Run training first: python plant_disease_detection_quick.py")
        return

    print("[INFO] Loading model ...")
    model = keras_models.load_model(MODEL_PATH)
    class_names = get_class_names()
    if class_names is None:
        print("[ERROR] Could not determine class names (no class_names.json / dataset).")
        return
    print(f"[INFO] Model loaded. {len(class_names)} classes:")
    for i, n in enumerate(class_names):
        print(f"   {i:2d}. {n}")

    import gradio as gr

    def predict(img):
        try:
            from PIL import Image
            # Gradio can hand us either a PIL Image or a numpy array
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            if img.mode != "RGB":
                img = img.convert("RGB")           # drop alpha / convert B/W
            arr = np.array(img)
            arr = kimage.smart_resize(arr, (224, 224))
            inp = np.expand_dims(arr, 0) / 255.0
            probs = model.predict(inp, verbose=0)[0]
            top = np.argsort(probs)[::-1]
            return {class_names[i]: float(probs[i]) for i in top[:5]}
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload a plant leaf image"),
        outputs=gr.Label(num_top_classes=5, label="Predicted Disease"),
        title="Plant Leaf Disease Detection",
        description=(
            "Upload a clear leaf image. The model identifies the crop disease "
            "or healthy condition. Trained on 7 classes for the IVA CIA-3 demo."
        ),
    )
    print("[INFO] Launching Gradio app at http://127.0.0.1:7860")
    demo.launch(share=False)


if __name__ == "__main__":
    main()