"""
Grad-CAM explainability — generates a heatmap showing which regions of the
X-ray activated the model's prediction.

Uses pytorch-grad-cam (pip install grad-cam) which handles hook registration,
gradient accumulation, and activation weighting automatically.

Why Grad-CAM matters for medical AI:
  Regulators (EU AI Act, FDA guidance) require medical AI to be interpretable.
  Grad-CAM lets a radiologist verify the model is looking at the right anatomy
  (e.g. lung consolidation, not the corner label).
"""
import numpy as np
import cv2
import torch
from PIL import Image
from torchvision import transforms
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

import config


def _get_preprocess() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(config.MEAN, config.STD),
    ])


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Invert ImageNet normalisation and return an HWC float32 in [0, 1]."""
    mean = np.array(config.MEAN)
    std  = np.array(config.STD)
    img  = tensor.cpu().numpy().transpose(1, 2, 0)
    img  = img * std + mean
    return np.clip(img, 0, 1).astype(np.float32)


def generate_gradcam(
    model: torch.nn.Module,
    image: Image.Image,
    device: torch.device,
) -> tuple[np.ndarray, int, float]:
    """
    Run inference and generate a Grad-CAM heatmap overlay.

    Args:
        model:  Loaded XRayClassifier in eval mode.
        image:  PIL image (any size, will be resized internally).
        device: CPU or CUDA.

    Returns:
        overlay:     RGB uint8 (224×224) heatmap fused with original image.
        pred_class:  0 = NORMAL, 1 = PNEUMONIA.
        confidence:  Softmax probability of the predicted class.
    """
    preprocess = _get_preprocess()
    input_tensor = preprocess(image).unsqueeze(0).to(device)

    # The last convolutional layer in EfficientNet-B0 sits at features[8][0]
    target_layer = [model.backbone.features[-1]]

    with GradCAM(model=model, target_layers=target_layer) as cam:
        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
        grayscale_cam = grayscale_cam[0]   # (H, W) in [0, 1]

    # Softmax confidence
    with torch.no_grad():
        logits = model(input_tensor)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_class = int(np.argmax(probs))
    confidence = float(probs[pred_class])

    # Build overlay
    rgb_img = _denormalize(input_tensor[0])
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    return overlay, pred_class, confidence


def side_by_side(original: Image.Image, overlay: np.ndarray) -> np.ndarray:
    """
    Combine the original resized X-ray and the Grad-CAM overlay side-by-side
    as a single uint8 RGB array — useful for README / paper figures.
    """
    orig = np.array(
        original.convert("RGB").resize((config.IMG_SIZE, config.IMG_SIZE))
    )
    gap = np.full((config.IMG_SIZE, 20, 3), 240, dtype=np.uint8)
    return np.concatenate([orig, gap, overlay], axis=1)
