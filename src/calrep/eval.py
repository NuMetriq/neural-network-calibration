from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metrics import compute_metrics, reliability_diagram_stats


@torch.no_grad()
def collect_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    model.eval()
    logits_list = []
    y_list = []

    for x, y in tqdm(loader, desc="infer", leave=False):
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        logits = model(x)
        logits_list.append(logits.detach().cpu())
        y_list.append(y.detach().cpu())

    logits_all = torch.cat(logits_list, dim=0)
    y_all = torch.cat(y_list, dim=0)
    return logits_all, y_all


def save_tensor(path: Path, t: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(t, path)


def save_json(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)


def evaluate_and_export(
    cfg: Dict[str, Any],
    model: nn.Module,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    ckpt_path: Path,
) -> Dict[str, Any]:
    out_root = Path(cfg.get("outputs", {}).get("root", "outputs"))
    report_dir = out_root / "reports"
    fig_dir = out_root / "figures"
    report_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Load checkpoint weights
    ckpt = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    n_bins = int(cfg.get("eval", {}).get("ece_bins", 15))
    include_brier = True

    # Collect val + test logits
    val_logits, val_y = collect_logits_and_labels(model, val_loader, device)
    test_logits, test_y = collect_logits_and_labels(model, test_loader, device)

    # Save tensors
    save_tensor(report_dir / "val_logits.pt", val_logits)
    save_tensor(report_dir / "val_labels.pt", val_y)
    save_tensor(report_dir / "test_logits.pt", test_logits)
    save_tensor(report_dir / "test_labels.pt", test_y)

    # Compute metrics (raw)
    val_metrics = compute_metrics(
        val_logits, val_y, n_bins=n_bins, include_brier=include_brier
    )
    test_metrics = compute_metrics(
        test_logits, test_y, n_bins=n_bins, include_brier=include_brier
    )

    raw = {
        "checkpoint": str(ckpt_path),
        "n_bins": n_bins,
        "val": val_metrics,
        "test": test_metrics,
    }
    save_json(report_dir / "metrics_raw.json", raw)

    # Save reliability stats arrays for plotting (JSON-friendly)
    val_rel = reliability_diagram_stats(val_logits, val_y, n_bins=n_bins)
    test_rel = reliability_diagram_stats(test_logits, test_y, n_bins=n_bins)
    rel_out = {
        "n_bins": n_bins,
        "val": {
            "bin_edges": val_rel.bin_edges.tolist(),
            "bin_counts": val_rel.bin_counts.tolist(),
            "bin_acc": [
                None if x != x else float(x) for x in val_rel.bin_acc
            ],  # NaN -> None
            "bin_conf": [None if x != x else float(x) for x in val_rel.bin_conf],
            "ece": val_rel.ece,
            "mce": val_rel.mce,
        },
        "test": {
            "bin_edges": test_rel.bin_edges.tolist(),
            "bin_counts": test_rel.bin_counts.tolist(),
            "bin_acc": [None if x != x else float(x) for x in test_rel.bin_acc],
            "bin_conf": [None if x != x else float(x) for x in test_rel.bin_conf],
            "ece": test_rel.ece,
            "mce": test_rel.mce,
        },
    }
    save_json(report_dir / "reliability_raw.json", rel_out)

    return raw
