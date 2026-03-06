from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np


def plot_ece_vs_bins(bins, raw_ece, temp_ece, title, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure()
    plt.plot(bins, raw_ece, marker="o")
    plt.plot(bins, temp_ece, marker="o")
    plt.xlabel("Number of bins")
    plt.ylabel("ECE")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_reliability(
    bin_edges: np.ndarray,
    bin_acc: np.ndarray,
    bin_conf: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert NaNs to mask
    mask = ~np.isnan(bin_acc) & ~np.isnan(bin_conf)
    acc = bin_acc[mask]
    conf = bin_conf[mask]
    edges = bin_edges

    # Use bin centers for bar positions
    bin_centers = (edges[:-1] + edges[1:]) / 2.0
    centers = bin_centers[mask]
    width = (edges[1] - edges[0]) * 0.9

    plt.figure()
    # Perfect calibration line
    plt.plot([0, 1], [0, 1])
    # Bars: accuracy per bin
    plt.bar(centers, acc, width=width, align="center", alpha=0.7, edgecolor="black")
    # Points: mean confidence per bin
    plt.scatter(centers, conf, marker="o")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
