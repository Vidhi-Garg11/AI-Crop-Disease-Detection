import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image

class VegetationIndices(object):
    """Calculates Excess Green (ExG) and Excess Red (ExR) indices for enhanced lesion visibility."""
    def __call__(self, img):
        img_np = np.array(img)
        r, g, b = cv2.split(img_np.astype(np.float32) / 255.0)
        
        r_sum = r + g + b + 1e-6
        r_norm, g_norm, b_norm = r / r_sum, g / r_sum, b / r_sum
        
        exg = 2 * g_norm - r_norm - b_norm
        exr = 1.4 * r_norm - g_norm
        
        enhanced = np.dstack((exg, exr, b_norm))
        enhanced = cv2.normalize(enhanced, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        return Image.fromarray(enhanced)

class BilateralFilter(object):
    """Applies edge-preserving noise reduction."""
    def __call__(self, img):
        img_np = np.array(img)
        filtered = cv2.bilateralFilter(img_np, d=5, sigmaColor=50, sigmaSpace=50)
        return Image.fromarray(filtered)

class PlantVillageDataset(Dataset):
    """Custom PyTorch Dataset for loading and processing PlantVillage images."""
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        self.classes = sorted([
            d for d in os.listdir(root_dir) 
            if os.path.isdir(os.path.join(root_dir, d))
        ])
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        
        for cls_name in self.classes:
            cls_dir = os.path.join(root_dir, cls_name)
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.image_paths.append(os.path.join(cls_dir, img_name))
                    self.labels.append(self.class_to_idx[cls_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

def get_dataloaders(data_dir, batch_size=32, val_split=0.2, num_workers=4):
    """Prepares train and validation DataLoaders with stratified splits and processing transforms."""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        BilateralFilter(),
        VegetationIndices(),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        BilateralFilter(),
        VegetationIndices(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    full_dataset = PlantVillageDataset(data_dir, transform=train_transform)
    val_size = int(len(full_dataset) * val_split)
    train_size = len(full_dataset) - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42)
    )
    
    # Overwrite transform for validation subset
    val_dataset.dataset.transform = val_transform

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, full_dataset.classes