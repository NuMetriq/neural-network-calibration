from __future__ import annotations

import torch.nn as nn
from torchvision.models import resnet18


def make_resnet18_cifar(num_classes: int) -> nn.Module:
    """
    Torchvision ResNet18 adapted for CIFAR-sized images (32x32):
    - replace first conv (3x3, stride 1)
    - remove maxpool
    - set final FC to num_classes
    """
    model = resnet18(weights=None)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def build_model(cfg: dict, num_classes: int) -> nn.Module:
    name = str(cfg.get("model", {}).get("name", "resnet18")).lower()
    if name != "resnet18":
        raise ValueError(f"Only resnet18 supported in v0.1.0, got {name}")
    return make_resnet18_cifar(num_classes=num_classes)
