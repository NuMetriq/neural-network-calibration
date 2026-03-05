import argparse
from pathlib import Path

import numpy as np
import torch
from calrep.data import get_cifar_dataloaders
from calrep.eval import evaluate_and_export
from calrep.models import build_model
from calrep.plotting import plot_reliability
from calrep.utils import load_config


def resolve_device(cfg):
    dev = str(cfg.get("device", "auto")).lower()
    if dev == "cpu":
        return torch.device("cpu")
    if dev == "cuda":
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", default="outputs/checkpoints/best.pt")
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = resolve_device(cfg)

    train_loader, val_loader, test_loader, meta = get_cifar_dataloaders(cfg)
    model = build_model(cfg, num_classes=meta.num_classes)

    raw = evaluate_and_export(
        cfg=cfg,
        model=model,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        ckpt_path=Path(args.ckpt),
    )

    # Plot reliability diagrams using the saved reliability json
    # (We already have raw logits, but this keeps plotting decoupled.)
    rel_path = (
        Path(cfg.get("outputs", {}).get("root", "outputs"))
        / "reports"
        / "reliability_raw.json"
    )
    rel = __import__("json").load(open(rel_path, "r", encoding="utf-8"))

    edges = np.array(rel["test"]["bin_edges"], dtype=float)
    acc = np.array(
        [np.nan if v is None else float(v) for v in rel["test"]["bin_acc"]], dtype=float
    )
    conf = np.array(
        [np.nan if v is None else float(v) for v in rel["test"]["bin_conf"]],
        dtype=float,
    )

    out_fig = (
        Path(cfg.get("outputs", {}).get("root", "outputs"))
        / "figures"
        / "reliability_raw.png"
    )
    plot_reliability(
        edges, acc, conf, title="Reliability Diagram (Raw, Test)", out_path=out_fig
    )

    print("Wrote metrics to outputs/reports/metrics_raw.json")
    print("Wrote reliability plot to", out_fig)
    print("Test metrics:", raw["test"])


if __name__ == "__main__":
    main()
