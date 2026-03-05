# data.py

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass(frozen=True)
class DataMeta:
    dataset_name: str
    num_classes: int
    train_size: int
    val_size: int
    test_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


_CIFAR_STATS = {
    "cifar10": {
        "mean": (0.4914, 0.4822, 0.4465),
        "std": (0.2023, 0.1994, 0.2010),
        "num_classes": 10,
        "ds": datasets.CIFAR10,
    },
    "cifar100": {
        "mean": (0.5071, 0.4867, 0.4408),
        "std": (0.2675, 0.2565, 0.2761),
        "num_classes": 100,
        "ds": datasets.CIFAR100,
    },
}


def _build_transforms(dataset_name: str):
    stats = _CIFAR_STATS[dataset_name]
    mean, std = stats["mean"], stats["std"]

    train_tf = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    eval_tf = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    return train_tf, eval_tf


def _deterministic_split_indices(
    n: int, val_fraction: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0,1), got {val_fraction}")

    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)

    val_size = int(round(n * val_fraction))
    val_idx = perm[:val_size]
    train_idx = perm[val_size:]

    # Sort indices for stable ordering (helps reproducibility in logging)
    return np.sort(train_idx), np.sort(val_idx)


def get_cifar_dataloaders(
    cfg: Dict[str, Any],
) -> Tuple[DataLoader, DataLoader, DataLoader, DataMeta]:
    """
    Returns train/val/test loaders from CIFAR-10 or CIFAR-100 with a deterministic train/val split.

    Expected cfg shape (flexible):
      cfg["dataset"]["name"] in {"cifar10","cifar100"}
      cfg["dataset"]["data_dir"] : path
      cfg["dataset"]["val_fraction"] : float
      cfg["dataset"]["num_workers"] : int
      batch_size either cfg["dataset"]["batch_size"] or cfg["train"]["batch_size"]
      seed either cfg["seed"] or cfg["train"]["seed"]
    """
    ds_cfg = cfg.get("dataset", {})
    dataset_name = str(ds_cfg.get("name", "cifar100")).lower()
    if dataset_name not in _CIFAR_STATS:
        raise ValueError(
            f"dataset.name must be one of {list(_CIFAR_STATS.keys())}, got {dataset_name}"
        )

    data_dir = Path(ds_cfg.get("data_dir", "data")).expanduser()
    val_fraction = float(ds_cfg.get("val_fraction", 0.1))
    num_workers = int(ds_cfg.get("num_workers", 4))

    seed = int(cfg.get("seed", cfg.get("train", {}).get("seed", 1337)))

    batch_size = ds_cfg.get("batch_size", None)
    if batch_size is None:
        batch_size = cfg.get("train", {}).get("batch_size", 128)
    batch_size = int(batch_size)

    train_tf, eval_tf = _build_transforms(dataset_name)
    ds_cls = _CIFAR_STATS[dataset_name]["ds"]

    # IMPORTANT: Create two dataset objects with different transforms.
    train_full = ds_cls(
        root=str(data_dir), train=True, transform=train_tf, download=True
    )
    eval_full = ds_cls(root=str(data_dir), train=True, transform=eval_tf, download=True)

    test_set = ds_cls(root=str(data_dir), train=False, transform=eval_tf, download=True)

    n = len(train_full)
    train_idx, val_idx = _deterministic_split_indices(
        n=n, val_fraction=val_fraction, seed=seed
    )

    train_set = Subset(train_full, train_idx.tolist())
    val_set = Subset(eval_full, val_idx.tolist())

    # Use a generator so shuffling is deterministic across runs (given same seed).
    g = torch.Generator()
    g.manual_seed(seed)

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        generator=g,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    meta = DataMeta(
        dataset_name=dataset_name,
        num_classes=_CIFAR_STATS[dataset_name]["num_classes"],
        train_size=len(train_set),
        val_size=len(val_set),
        test_size=len(test_set),
        mean=_CIFAR_STATS[dataset_name]["mean"],
        std=_CIFAR_STATS[dataset_name]["std"],
    )

    return train_loader, val_loader, test_loader, meta
