import argparse
import random

import numpy as np
import torch
from calrep.data import get_cifar_dataloaders
from calrep.models import build_model
from calrep.train import train_model
from calrep.utils import load_config


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def resolve_device(cfg):
    dev = str(cfg.get("device", "auto")).lower()
    if dev == "cpu":
        return torch.device("cpu")
    if dev == "cuda":
        return torch.device("cuda")
    # auto
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    args = ap.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg.get("seed", 1337))
    set_seed(seed)
    device = resolve_device(cfg)

    train_loader, val_loader, test_loader, meta = get_cifar_dataloaders(cfg)
    model = build_model(cfg, num_classes=meta.num_classes)

    print(
        f"Dataset={meta.dataset_name} classes={meta.num_classes} sizes="
        f"{meta.train_size}/{meta.val_size}/{meta.test_size}"
    )
    print("Device:", device)

    best_path = train_model(cfg, model, train_loader, val_loader, device)
    print("Saved best checkpoint to:", best_path)


if __name__ == "__main__":
    main()
