"""
Model definition — EfficientNet-B0 with a custom classification head.

Transfer learning strategy:
  Phase 1: freeze all backbone weights, train only the head (5 epochs).
  Phase 2: unfreeze the last two blocks + head, fine-tune with lower LR.

This two-phase approach prevents catastrophic forgetting while adapting
ImageNet features to the narrow domain of grayscale medical images.
"""
import torch
import torch.nn as nn
from torchvision import models
import config


class XRayClassifier(nn.Module):
    """EfficientNet-B0 backbone with a dropout-regularised binary head."""

    def __init__(self, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.DEFAULT if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)

        # Replace the default classifier with a stronger head
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.4, inplace=True),
            nn.Linear(in_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

    def freeze_backbone(self):
        """Freeze all backbone layers except the classifier head."""
        for name, param in self.backbone.named_parameters():
            if "classifier" not in name:
                param.requires_grad = False
        print("Backbone frozen — training head only.")

    def unfreeze_top_blocks(self, num_blocks: int = 2):
        """
        Unfreeze the last `num_blocks` feature blocks for fine-tuning.
        EfficientNet-B0 has 9 feature blocks (features.0 … features.8).
        """
        trainable = set()
        for name, param in self.backbone.named_parameters():
            if "classifier" in name:
                param.requires_grad = True
                trainable.add(name)
            else:
                block_id = None
                parts = name.split(".")
                if parts[0] == "features" and parts[1].isdigit():
                    block_id = int(parts[1])
                if block_id is not None and block_id >= (9 - num_blocks):
                    param.requires_grad = True
                    trainable.add(name)
                else:
                    param.requires_grad = False
        n = sum(p.requires_grad for p in self.backbone.parameters())
        print(f"Unfrozen top {num_blocks} blocks — {n} trainable params.")

    def count_parameters(self):
        total   = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total params: {total:,} | Trainable: {trainable:,}")


def load_model(path: str, device: torch.device) -> XRayClassifier:
    """Load a saved checkpoint and return the model in eval mode."""
    model = XRayClassifier(pretrained=False)
    state = torch.load(path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.eval()
    print(f"Model loaded from {path} (val_acc={state.get('val_acc', '?'):.4f})")
    return model
