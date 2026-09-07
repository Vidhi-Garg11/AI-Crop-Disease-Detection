"""
Tests for Grad-CAM Module
===========================

Tests the GradCAM class with a small ResNet18 model and random inputs
to verify heatmap generation, overlay blending, and target layer detection.
"""

import os
import sys
import tempfile

import numpy as np
import pytest
import torch
from PIL import Image
from torchvision.models import resnet18

# Add src/ to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from xai_module.gradcam import GradCAM, get_target_layer


@pytest.fixture
def resnet_model():
    """Create a small ResNet18 for testing (untrained, random weights)."""
    model = resnet18(weights=None, num_classes=10)
    model.eval()
    return model


@pytest.fixture
def random_input():
    """Create a random input tensor of shape (1, 3, 224, 224)."""
    torch.manual_seed(42)
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def random_image():
    """Create a random RGB image as a PIL Image."""
    np.random.seed(42)
    arr = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    return Image.fromarray(arr)


class TestGradCAMGeneration:
    """Tests for heatmap generation."""

    def test_heatmap_shape(self, resnet_model, random_input):
        """Heatmap should have spatial dimensions of the target layer's output."""
        target_layer = resnet_model.layer4[-1]
        gradcam = GradCAM(resnet_model, target_layer)

        heatmap = gradcam.generate(random_input)
        gradcam.remove_hooks()

        # ResNet18 layer4 output is 7x7 for 224x224 input
        assert heatmap.shape == (7, 7), f"Expected (7, 7), got {heatmap.shape}"

    def test_heatmap_value_range(self, resnet_model, random_input):
        """Heatmap values should be normalized to [0, 1]."""
        target_layer = resnet_model.layer4[-1]
        gradcam = GradCAM(resnet_model, target_layer)

        heatmap = gradcam.generate(random_input)
        gradcam.remove_hooks()

        assert heatmap.min() >= 0.0, f"Min value {heatmap.min()} < 0"
        assert heatmap.max() <= 1.0, f"Max value {heatmap.max()} > 1"

    def test_specific_target_class(self, resnet_model, random_input):
        """Should generate heatmap for a specific target class."""
        target_layer = resnet_model.layer4[-1]
        gradcam = GradCAM(resnet_model, target_layer)

        heatmap = gradcam.generate(random_input, target_class=3)
        gradcam.remove_hooks()

        assert heatmap.shape == (7, 7)
        assert heatmap.min() >= 0.0
        assert heatmap.max() <= 1.0

    def test_invalid_input_shape(self, resnet_model):
        """Should raise ValueError for non-batch input."""
        target_layer = resnet_model.layer4[-1]
        gradcam = GradCAM(resnet_model, target_layer)

        with pytest.raises(ValueError, match="Expected input_tensor of shape"):
            gradcam.generate(torch.randn(3, 224, 224))

        gradcam.remove_hooks()


class TestGradCAMOverlay:
    """Tests for overlay blending."""

    def test_overlay_shape_pil(self, resnet_model, random_input, random_image):
        """Overlay should match the original image dimensions."""
        target_layer = resnet_model.layer4[-1]
        gradcam = GradCAM(resnet_model, target_layer)

        heatmap = gradcam.generate(random_input)
        overlay = gradcam.overlay(heatmap, random_image)
        gradcam.remove_hooks()

        assert overlay.shape == (224, 224, 3), f"Got shape {overlay.shape}"
        assert overlay.dtype == np.uint8

    def test_overlay_shape_numpy(self, resnet_model, random_input):
        """Overlay should work with numpy array images too."""
        target_layer = resnet_model.layer4[-1]
        gradcam = GradCAM(resnet_model, target_layer)

        heatmap = gradcam.generate(random_input)
        np_image = np.random.randint(0, 255, (100, 150, 3), dtype=np.uint8)
        overlay = gradcam.overlay(heatmap, np_image)
        gradcam.remove_hooks()

        assert overlay.shape == (100, 150, 3)


class TestGradCAMSaveVisualization:
    """Tests for saving visualizations to disk."""

    def test_save_creates_files(self, resnet_model, random_input, random_image):
        """Should save original, heatmap, and overlay images."""
        target_layer = resnet_model.layer4[-1]
        gradcam = GradCAM(resnet_model, target_layer)

        heatmap = gradcam.generate(random_input)
        overlay = gradcam.overlay(heatmap, random_image)
        gradcam.remove_hooks()

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = GradCAM.save_visualization(
                random_image, heatmap, overlay, tmpdir, prefix="test"
            )

            assert os.path.exists(paths["original"])
            assert os.path.exists(paths["heatmap"])
            assert os.path.exists(paths["overlay"])


class TestGetTargetLayer:
    """Tests for auto-detecting target layers."""

    def test_resnet_detection(self, resnet_model):
        """Should detect ResNet's layer4[-1]."""
        layer = get_target_layer(resnet_model)
        assert layer is resnet_model.layer4[-1]

    def test_explicit_hint(self, resnet_model):
        """Should work with explicit architecture hint."""
        layer = get_target_layer(resnet_model, architecture_hint="resnet")
        assert layer is resnet_model.layer4[-1]

    def test_unknown_architecture(self):
        """Should raise ValueError for unknown architectures."""
        model = torch.nn.Linear(10, 5)
        with pytest.raises(ValueError, match="Cannot auto-detect"):
            get_target_layer(model)
