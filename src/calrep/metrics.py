from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class ReliabilityResult:
    bin_edges: np.ndarray  # shape (M+1,)
    bin_counts: np.ndarray  # shape (M,)
    bin_acc: np.ndarray  # shape (M,)
    bin_conf: np.ndarray  # shape (M,)
    ece: float
    mce: float


def accuracy_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    preds = torch.argmax(logits, dim=1)
    return float((preds == y).float().mean().item())


def nll_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    # mean NLL (cross-entropy)
    return float(F.cross_entropy(logits, y).item())


def brier_multiclass_from_logits(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Multiclass Brier score (mean squared error between probs and one-hot labels).
    """
    probs = torch.softmax(logits, dim=1)
    num_classes = probs.shape[1]
    y_onehot = F.one_hot(y, num_classes=num_classes).float()
    return float(torch.mean(torch.sum((probs - y_onehot) ** 2, dim=1)).item())


def reliability_diagram_stats(
    logits: torch.Tensor,
    y: torch.Tensor,
    n_bins: int = 15,
) -> ReliabilityResult:
    """
    Equal-width binning on confidence in [0,1].
    Returns per-bin accuracy and confidence plus ECE/MCE.

    This matches the standard definition used in Guo et al. (ECE with M=15 bins).
    """
    if logits.ndim != 2:
        raise ValueError(f"logits must be 2D (N,K), got shape {tuple(logits.shape)}")
    if y.ndim != 1:
        raise ValueError(f"y must be 1D (N,), got shape {tuple(y.shape)}")

    probs = torch.softmax(logits, dim=1)
    conf, preds = torch.max(probs, dim=1)
    correct = (preds == y).float()

    conf_np = conf.detach().cpu().numpy()
    correct_np = correct.detach().cpu().numpy()

    # Bin edges include 0 and 1.
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    # Assign each confidence to a bin index in [0, n_bins-1]
    # np.digitize gives 1..n_bins (right=False), so subtract 1.
    bin_ids = np.digitize(conf_np, bin_edges[1:-1], right=False)

    bin_counts = np.zeros(n_bins, dtype=np.int64)
    bin_acc = np.zeros(n_bins, dtype=np.float64)
    bin_conf = np.zeros(n_bins, dtype=np.float64)

    n = conf_np.shape[0]
    ece = 0.0
    mce = 0.0

    for b in range(n_bins):
        mask = bin_ids == b
        cnt = int(mask.sum())
        bin_counts[b] = cnt
        if cnt == 0:
            bin_acc[b] = np.nan
            bin_conf[b] = np.nan
            continue

        acc_b = float(correct_np[mask].mean())
        conf_b = float(conf_np[mask].mean())
        bin_acc[b] = acc_b
        bin_conf[b] = conf_b

        gap = abs(acc_b - conf_b)
        ece += (cnt / n) * gap
        mce = max(mce, gap)

    return ReliabilityResult(
        bin_edges=bin_edges,
        bin_counts=bin_counts,
        bin_acc=bin_acc,
        bin_conf=bin_conf,
        ece=float(ece),
        mce=float(mce),
    )


def compute_metrics(
    logits: torch.Tensor,
    y: torch.Tensor,
    n_bins: int = 15,
    include_brier: bool = True,
) -> Dict[str, float]:
    rel = reliability_diagram_stats(logits, y, n_bins=n_bins)
    out = {
        "accuracy": accuracy_from_logits(logits, y),
        "nll": nll_from_logits(logits, y),
        "ece": rel.ece,
        "mce": rel.mce,
        "ece_bins": float(n_bins),
    }
    if include_brier:
        out["brier"] = brier_multiclass_from_logits(logits, y)
    return out
