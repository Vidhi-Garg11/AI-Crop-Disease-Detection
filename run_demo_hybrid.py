"""
=====================================================================
 Hybrid-Backend Gradio Web Demo for Plant Leaf Disease Detection
=====================================================================
Serves predictions from the pretrained MobileNetV3-Large + Vision
Transformer checkpoint (25 PlantVillage classes). Preprocessing
matches the model: bilateral filter -> vegetation indices ->
ImageNet normalization. See hybrid/hybrid_backend.py.

Usage:
    python run_demo_hybrid.py
=====================================================================
"""

import os
import sys

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(PROJECT_DIR, "hybrid"))

import hybrid_backend as hb  # noqa: E402


def main():
    import torch
    print("[INFO] Loading hybrid checkpoint (torch %s) ..." % torch.__version__)
    model, classes, val_acc = hb.load_hybrid()
    print("[INFO] Checkpoint: stored val_acc = %.2f%%, %d classes"
          % (val_acc * 100, len(classes)))
    for i, n in enumerate(classes):
        print(f"   {i:2d}. {n}")

    import gradio as gr

    def predict(img):
        try:
            from PIL import Image
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            if img.mode != "RGB":
                img = img.convert("RGB")
            res = hb.predict_pil(img, topk=5)
            return {clean(label): prob for label, prob in res}
        except Exception as e:
            return {"error": f"Prediction failed: {str(e)}"}

    def clean(label):
        return label.replace("___", " · ").replace("_", " ").replace("(including sour)", "")

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil", label="Upload a plant leaf image"),
        outputs=gr.Label(num_top_classes=5, label="Predicted Disease"),
        title="Plant Leaf Disease Detection — Hybrid Backend",
        description=(
            "MobileNetV3-Large + Vision-Transformer hybrid (25 PlantVillage "
            "classes). Preprocessing: bilateral filter + vegetation indices. "
            "Trained by the AI-Crop-Disease-Detection project "
            "(github.com/Vidhi-Garg11/AI-Crop-Disease-Detection)."
        ),
    )
    print("[INFO] Launching Gradio app at http://127.0.0.1:%s" %
          os.environ.get("PORT", "7860"))
    demo.launch(share=False, show_error=True,
                server_port=int(os.environ.get("PORT", "7860")))


if __name__ == "__main__":
    main()