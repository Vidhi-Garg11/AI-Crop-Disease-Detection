"""
=====================================================================
 Automatic Dataset Downloader (Kaggle) - Plant Leaf Disease
=====================================================================
Downloads the 'New Plant Diseases Dataset' from Kaggle using kagglehub
and copies train/ + valid/ straight into this project's 'dataset/'.

Requirements:
    pip install kagglehub

Usage:
    python download_dataset.py
=====================================================================
"""

import os
import shutil
import sys

try:
    import kagglehub
except ImportError:
    print("[ERROR] kagglehub not installed. Run:  pip install kagglehub")
    sys.exit(1)

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
DEST_DIR    = os.path.join(PROJECT_DIR, "dataset")
DATASET_SLUG = "vipoooool/new-plant-diseases-dataset"


def find_subfolders(root, name):
    """Return all directories named `name` found anywhere under `root`."""
    hits = []
    for dirpath, dirnames, filenames in os.walk(root):
        if name in dirnames:
            hits.append(os.path.join(dirpath, name))
    return hits


def count_images(folder):
    n = 0
    for _, _, files in os.walk(folder):
        n += len([f for f in files if f.lower().endswith((".jpg", ".jpeg", ".png"))])
    return n


def main():
    print("[1/4] Downloading dataset from Kaggle ...")
    path = kagglehub.dataset_download(DATASET_SLUG)
    print(f"      Downloaded / cached at: {path}")

    os.makedirs(DEST_DIR, exist_ok=True)

    copied = False
    for sub in ("train", "valid", "test"):
        hits = find_subfolders(path, sub)
        if not hits:
            print(f"      [WARN] No '{sub}/' folder found under cache")
            continue
        src = hits[0]
        dst = os.path.join(DEST_DIR, sub)
        if os.path.isdir(dst):
            n = count_images(dst)
            print(f"      [SKIP] {dst} already exists ({n} images)")
        else:
            n = count_images(src)
            print(f"[2/4] Copying '{sub}/' ({n} images) -> {dst} ...")
            shutil.copytree(src, dst)
        copied = True

    if not copied:
        print("[ERROR] Could not find train/valid/test folders. Aborting.")
        sys.exit(1)

    print("[3/4] Verifying ...")
    for sub in ("train", "valid"):
        folder = os.path.join(DEST_DIR, sub)
        if os.path.isdir(folder):
            classes = len(os.listdir(folder))
            imgs = count_images(folder)
            print(f"      {sub:5s}: {classes} classes, {imgs} images")

    print("[4/4] Done!")
    print(f"      Dataset is ready at: {DEST_DIR}")
    print("      Run:  python plant_disease_detection.py")


if __name__ == "__main__":
    main()