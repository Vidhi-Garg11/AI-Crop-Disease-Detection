import os
import time
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from hybrid_model import MobileNetViTHybrid
from data_loader import BilateralFilter, VegetationIndices

def get_inference_transforms():
    """Applies the exact same preprocessing used during validation."""
    return transforms.Compose([
        transforms.Resize((224, 224)),
        BilateralFilter(),
        VegetationIndices(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def print_model_efficiency(model, checkpoint_path):
    """Calculates model size and parameter count for edge deployment validation."""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    file_size_mb = os.path.getsize(checkpoint_path) / (1024 * 1024)
    
    print("\n" + "="*40)
    print(" 📊 EFFICIENCY BENCHMARK")
    print("="*40)
    print(f"Total Parameters:     {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Checkpoint Size:      {file_size_mb:.2f} MB")
    print("="*40)

def predict_and_benchmark(model, image_path, classes, device):
    """Runs a forward pass and measures latency accurately."""
    transform = get_inference_transforms()
    
    try:
        image = Image.open(image_path).convert("RGB")
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return

    # Add batch dimension (B, C, H, W)
    input_tensor = transform(image).unsqueeze(0).to(device)

    # --- 1. WARM-UP PHASE ---
    # GPU initialization can take up to 3 seconds. We run dummy inputs to wake it up.
    if device.type == 'cuda':
        for _ in range(5):
            _ = model(input_tensor)
            
    # --- 2. ACCURATE LATENCY MEASUREMENT ---
    model.eval()
    with torch.no_grad():
        if device.type == 'cuda':
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            
            start_event.record()
            with torch.amp.autocast('cuda'):
                logits = model(input_tensor)
            end_event.record()
            
            # Wait for GPU synchronization before calculating time
            torch.cuda.synchronize()
            inference_time_ms = start_event.elapsed_time(end_event)
        else:
            start_time = time.time()
            logits = model(input_tensor)
            inference_time_ms = (time.time() - start_time) * 1000

    # Calculate probabilities and predicted class
    probabilities = F.softmax(logits, dim=1)[0]
    confidence_score, predicted_idx = torch.max(probabilities, dim=0)
    predicted_class = classes[predicted_idx.item()]

    print("\n" + "="*40)
    print(" 🌿 INFERENCE RESULTS")
    print("="*40)
    print(f"Target Image:      {os.path.basename(image_path)}")
    print(f"Predicted Disease: {predicted_class}")
    print(f"Confidence Score:  {confidence_score.item() * 100:.2f}%")
    print(f"Inference Latency: {inference_time_ms:.2f} ms")
    print(f"Frames Per Second: {1000 / inference_time_ms:.1f} FPS")
    print("="*40 + "\n")

def main():
    CHECKPOINT_PATH = "../checkpoints/hybrid_model_best.pth"
    # Provide a path to a real-world test image (e.g., from PlantDoc or your smartphone)
    TEST_IMAGE_PATH = "../data/test_image.JPG" 

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Running inference on: {device}")

    if not os.path.exists(CHECKPOINT_PATH):
        print(f"[ERROR] Checkpoint not found at {CHECKPOINT_PATH}")
        return

    # Load Checkpoint and Rebuild Model
    print("[INFO] Loading trained model weights...")
    checkpoint = torch.load(CHECKPOINT_PATH, map_location=device)
    classes = checkpoint['classes']
    
    model = MobileNetViTHybrid(num_classes=len(classes)).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Run the efficiency benchmark
    print_model_efficiency(model, CHECKPOINT_PATH)

    # Run inference on the target image
    if os.path.exists(TEST_IMAGE_PATH):
        predict_and_benchmark(model, TEST_IMAGE_PATH, classes, device)
    else:
        print(f"[WARNING] Please place a test image at: {TEST_IMAGE_PATH}")

if __name__ == "__main__":
    main()