import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
from data_loader import get_dataloaders
from hybrid_model import MobileNetViTHybrid

def evaluate_full_metrics():
    DATA_DIR = "../data/raw/color"
    CHECKPOINT_PATH = "../checkpoints/hybrid_model_best.pth"
    BATCH_SIZE = 32

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Evaluating on device: {device}")

    # Load dataloaders and checkpoint
    _, val_loader, classes = get_dataloaders(DATA_DIR, batch_size=BATCH_SIZE)
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)

    model = MobileNetViTHybrid(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()

    all_preds = []
    all_targets = []

    print("[INFO] Collecting predictions across validation set...")
    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc="Evaluating"):
            images = images.to(device)
            with torch.amp.autocast('cuda'):
                logits = model(images)
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            
            all_preds.extend(preds)
            all_targets.extend(labels.numpy())

    # Print summary metrics
    print("\n" + "="*50)
    print(" 📈 FINAL MODEL CLASSIFICATION REPORT")
    print("="*50)
    print(classification_report(all_targets, all_preds, target_names=classes, digits=4))

if __name__ == "__main__":
    evaluate_full_metrics()