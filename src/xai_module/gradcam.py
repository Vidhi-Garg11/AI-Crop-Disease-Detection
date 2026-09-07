"""
Grad-CAM — Gradient-weighted Class Activation Mapping
======================================================

Architecture-agnostic Grad-CAM implementation for PyTorch CNN models.
Generates visual explanations showing which regions of the input image
most influenced the model's classification decision.

Supports:
- ResNet (torchvision)
- EfficientNet (torchvision)
- MobileNetV2 / MobileNetV3 (torchvision)
- MobileNetViTHybrid (project's custom hybrid model)
- Any PyTorch model with convolutional layers

Reference:
    Selvaraju et al., "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization", ICCV 2017.

Author: Vidhi Garg
"""

import os
from typing import Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


class GradCAM:
    """
    Gradient-weighted Class Activation Mapping (Grad-CAM).

    Computes a coarse localization heatmap highlighting important regions
    in the input image for a given target class prediction.

    Usage:
        >>> gradcam = GradCAM(model, target_layer=model.layer4[-1])
        >>> heatmap = gradcam.generate(input_tensor, target_class=3)
        >>> overlay = gradcam.overlay(heatmap, original_image)

    Args:
        model: A PyTorch nn.Module (must be in eval mode for meaningful results).
        target_layer: The convolutional layer (nn.Module) to extract activations from.
                      Typically the last convolutional layer before the classifier.
    """

    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer

        # Storage for hooked activations and gradients
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None

        # Register hooks on the target layer
        self._forward_hook = target_layer.register_forward_hook(self._save_activation)
        self._backward_hook = target_layer.register_full_backward_hook(
            self._save_gradient
        )

    def _save_activation(
        self,
        module: torch.nn.Module,
        input: Tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        """Forward hook: captures the activation output of the target layer."""
        self._activations = output.detach()

    def _save_gradient(
        self,
        module: torch.nn.Module,
        grad_input: Tuple[torch.Tensor, ...],
        grad_output: Tuple[torch.Tensor, ...],
    ) -> None:
        """Backward hook: captures the gradient flowing into the target layer."""
        self._gradients = grad_output[0].detach()

    def generate(
        self,
        input_tensor: torch.Tensor,
        target_class: Optional[int] = None,
    ) -> np.ndarray:
        """
        Generate a Grad-CAM heatmap for the given input.

        Args:
            input_tensor: Preprocessed input tensor of shape (1, C, H, W).
            target_class: The class index to generate the heatmap for.
                          If None, uses the predicted (argmax) class.

        Returns:
            heatmap: Normalized heatmap as a numpy array of shape (H_feat, W_feat),
                     values in [0, 1]. This is at the spatial resolution of the
                     target layer's feature map.
        """
        if input_tensor.dim() != 4 or input_tensor.size(0) != 1:
            raise ValueError(
                f"Expected input_tensor of shape (1, C, H, W), got {input_tensor.shape}"
            )

        # Ensure model is in eval mode for deterministic behavior
        self.model.eval()

        # Forward pass — hooks capture activations
        output = self.model(input_tensor)

        # Determine target class
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # Zero all existing gradients
        self.model.zero_grad()

        # Backward pass for the target class score
        # We backprop the score for the target class (before softmax)
        target_score = output[0, target_class]
        target_score.backward(retain_graph=False)

        # Validate that hooks captured data
        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "Grad-CAM hooks did not capture activations/gradients. "
                "Verify that the target_layer is part of the model's forward pass."
            )

        # Grad-CAM computation:
        # 1. Global average pool the gradients over spatial dimensions → channel weights
        # 2. Weighted combination of activation maps
        # 3. ReLU to keep only positive contributions

        # gradients shape: (1, C, H, W)
        weights = self._gradients.mean(dim=(2, 3), keepdim=True)  # (1, C, 1, 1)

        # Weighted sum of activations
        cam = (weights * self._activations).sum(dim=1, keepdim=True)  # (1, 1, H, W)

        # ReLU — we only care about features that positively influence the target class
        cam = F.relu(cam)

        # Remove batch and channel dimensions
        cam = cam.squeeze().cpu().numpy()  # (H, W)

        # Normalize to [0, 1]
        cam_min, cam_max = cam.min(), cam.max()
        if cam_max - cam_min > 1e-8:
            cam = (cam - cam_min) / (cam_max - cam_min)
        else:
            cam = np.zeros_like(cam)

        return cam

    def overlay(
        self,
        heatmap: np.ndarray,
        original_image: Union[np.ndarray, Image.Image],
        alpha: float = 0.5,
        colormap: int = cv2.COLORMAP_JET,
    ) -> np.ndarray:
        """
        Overlay the Grad-CAM heatmap onto the original image.

        Args:
            heatmap: Normalized heatmap (H, W) with values in [0, 1].
            original_image: Original RGB image (PIL Image or numpy array).
            alpha: Blending factor (0 = only image, 1 = only heatmap).
            colormap: OpenCV colormap to apply to the heatmap.

        Returns:
            overlay: Blended image as a numpy array (H, W, 3) in RGB format, uint8.
        """
        # Convert PIL to numpy if needed
        if isinstance(original_image, Image.Image):
            original_image = np.array(original_image)

        img_h, img_w = original_image.shape[:2]

        # Resize heatmap to match original image dimensions
        heatmap_resized = cv2.resize(heatmap, (img_w, img_h))

        # Apply colormap (OpenCV uses BGR, we convert to RGB)
        heatmap_colored = cv2.applyColorMap(
            np.uint8(255 * heatmap_resized), colormap
        )
        heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

        # Blend
        overlay = np.uint8(alpha * heatmap_colored + (1 - alpha) * original_image)

        return overlay

    @staticmethod
    def save_visualization(
        original_image: Union[np.ndarray, Image.Image],
        heatmap: np.ndarray,
        overlay: np.ndarray,
        save_dir: str,
        prefix: str = "gradcam",
    ) -> dict:
        """
        Save the original image, heatmap, and overlay to disk.

        Args:
            original_image: Original RGB image.
            heatmap: Raw heatmap array (H, W) in [0, 1].
            overlay: Blended overlay image (H, W, 3).
            save_dir: Directory to save outputs.
            prefix: Filename prefix.

        Returns:
            Dictionary with saved file paths.
        """
        os.makedirs(save_dir, exist_ok=True)

        paths = {}

        # Save original
        orig_path = os.path.join(save_dir, f"{prefix}_original.png")
        if isinstance(original_image, Image.Image):
            original_image.save(orig_path)
        else:
            Image.fromarray(original_image).save(orig_path)
        paths["original"] = orig_path

        # Save heatmap
        heatmap_path = os.path.join(save_dir, f"{prefix}_heatmap.png")
        fig, ax = plt.subplots(1, 1, figsize=(6, 6))
        ax.imshow(heatmap, cmap="jet", interpolation="bilinear")
        ax.set_title("Grad-CAM Heatmap")
        ax.axis("off")
        fig.savefig(heatmap_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        paths["heatmap"] = heatmap_path

        # Save overlay
        overlay_path = os.path.join(save_dir, f"{prefix}_overlay.png")
        Image.fromarray(overlay).save(overlay_path)
        paths["overlay"] = overlay_path

        return paths

    def remove_hooks(self) -> None:
        """Remove registered hooks to free memory."""
        self._forward_hook.remove()
        self._backward_hook.remove()

    def __del__(self):
        """Cleanup hooks on garbage collection."""
        try:
            self.remove_hooks()
        except Exception:
            pass


def get_target_layer(
    model: torch.nn.Module, architecture_hint: Optional[str] = None
) -> torch.nn.Module:
    """
    Auto-detect the appropriate target convolutional layer for Grad-CAM.

    Supports the following architectures:
    - "hybrid" / "MobileNetViTHybrid" → model.feature_extractor[-1]
    - "resnet" → model.layer4[-1]
    - "efficientnet" → model.features[-1]
    - "mobilenet" → model.features[-1]

    If architecture_hint is None, attempts to detect from the model class name.

    Args:
        model: The PyTorch model.
        architecture_hint: Optional string hint for the architecture type.

    Returns:
        The target nn.Module (convolutional layer) for Grad-CAM.

    Raises:
        ValueError: If the architecture cannot be detected or is unsupported.
    """
    # Auto-detect from class name if no hint provided
    if architecture_hint is None:
        class_name = model.__class__.__name__.lower()
        if "hybrid" in class_name or "mobilenetvit" in class_name:
            architecture_hint = "hybrid"
        elif "resnet" in class_name:
            architecture_hint = "resnet"
        elif "efficientnet" in class_name:
            architecture_hint = "efficientnet"
        elif "mobilenet" in class_name:
            architecture_hint = "mobilenet"

    hint = (architecture_hint or "").lower()

    try:
        if hint == "hybrid":
            # MobileNetViTHybrid: last block of MobileNetV3 feature extractor
            return model.feature_extractor[-1]
        elif hint == "resnet":
            return model.layer4[-1]
        elif hint == "efficientnet":
            return model.features[-1]
        elif hint in ("mobilenet", "mobilenetv2", "mobilenetv3"):
            return model.features[-1]
        else:
            raise ValueError(
                f"Cannot auto-detect target layer for architecture '{architecture_hint}'. "
                f"Model class: {model.__class__.__name__}. "
                f"Please provide the target layer explicitly, e.g.: "
                f"GradCAM(model, target_layer=model.layer4[-1])"
            )
    except AttributeError as e:
        raise ValueError(
            f"Failed to access expected layer for '{architecture_hint}': {e}. "
            f"Please provide the target layer explicitly."
        ) from e
