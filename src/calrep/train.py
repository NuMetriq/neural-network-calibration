from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader
from tqdm import tqdm


def _accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return (preds == y).float().mean().item()


@torch.no_grad()
def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device
) -> Dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        logits = model(x)
        loss = F.cross_entropy(logits, y, reduction="sum")

        total_loss += loss.item()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += y.numel()

    nll = total_loss / total_n
    acc = total_correct / total_n
    return {"nll": float(nll), "acc": float(acc)}


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in tqdm(loader, desc="train", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * y.numel()
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += y.numel()

    nll = total_loss / total_n
    acc = total_correct / total_n
    return {"nll": float(nll), "acc": float(acc)}


def train_model(
    cfg: Dict[str, Any],
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
) -> Path:
    out_root = Path(cfg.get("outputs", {}).get("root", "outputs"))
    ckpt_dir = out_root / "checkpoints"
    report_dir = out_root / "reports"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    log_path = report_dir / "train_log.jsonl"

    train_cfg = cfg.get("train", {})
    epochs = int(train_cfg.get("epochs", 50))
    lr = float(train_cfg.get("lr", 0.1))
    momentum = float(train_cfg.get("momentum", 0.9))
    weight_decay = float(train_cfg.get("weight_decay", 5e-4))

    optimizer = SGD(
        model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay
    )

    best_acc = -1.0
    best_path = ckpt_dir / "best.pt"

    # Optional: torch.backends settings for reproducibility-ish
    # (True determinism is more involved; we’ll address later if needed.)
    model.to(device)

    with log_path.open("w", encoding="utf-8") as f:
        for epoch in range(1, epochs + 1):
            train_metrics = train_one_epoch(model, train_loader, optimizer, device)
            val_metrics = evaluate(model, val_loader, device)

            row = {
                "epoch": epoch,
                "train_nll": train_metrics["nll"],
                "train_acc": train_metrics["acc"],
                "val_nll": val_metrics["nll"],
                "val_acc": val_metrics["acc"],
                "lr": lr,
            }
            f.write(json.dumps(row) + "\n")
            f.flush()

            if val_metrics["acc"] > best_acc:
                best_acc = val_metrics["acc"]
                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "cfg": cfg,
                        "best_val_acc": best_acc,
                        "epoch": epoch,
                    },
                    best_path,
                )

            print(
                f"Epoch {epoch:03d} | "
                f"train acc {train_metrics['acc']:.4f} nll {train_metrics['nll']:.4f} | "
                f"val acc {val_metrics['acc']:.4f} nll {val_metrics['nll']:.4f} | "
                f"best {best_acc:.4f}"
            )

    return best_path
