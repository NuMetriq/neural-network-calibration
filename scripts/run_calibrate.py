import argparse
import json
from pathlib import Path

import numpy as np
import torch
from calrep.calibrate import apply_temperature, fit_temperature
from calrep.metrics import compute_metrics, reliability_diagram_stats
from calrep.plotting import plot_reliability
from calrep.utils import load_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--reports_dir", default="outputs/reports")
    ap.add_argument("--figures_dir", default="outputs/figures")
    args = ap.parse_args()

    cfg = load_config(args.config)
    n_bins = int(cfg.get("eval", {}).get("ece_bins", 15))

    reports_dir = Path(args.reports_dir)
    figures_dir = Path(args.figures_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load saved logits/labels (CPU is fine)
    val_logits = torch.load(reports_dir / "val_logits.pt", map_location="cpu")
    assert isinstance(val_logits, torch.Tensor)
    val_y = torch.load(reports_dir / "val_labels.pt", map_location="cpu")
    test_logits = torch.load(reports_dir / "test_logits.pt", map_location="cpu")
    test_y = torch.load(reports_dir / "test_labels.pt", map_location="cpu")

    # Sanity: accuracy must be unchanged after temperature scaling
    raw_preds = test_logits.argmax(dim=1)

    # Fit temperature on validation set
    fit = fit_temperature(val_logits, val_y)

    # Apply to val/test
    val_logits_t = apply_temperature(val_logits, fit.temperature)
    test_logits_t = apply_temperature(test_logits, fit.temperature)

    from calrep.metrics import ece_sweep

    raw = json.load(open(reports_dir / "metrics_raw.json", "r", encoding="utf-8"))

    ece_bins = [5, 10, 15, 20, 30, 50]
    raw_ece = ece_sweep(test_logits, test_y, bins_list=ece_bins)
    temp_ece = ece_sweep(test_logits_t, test_y, bins_list=ece_bins)

    sweep_out = {
        "bins": ece_bins,
        "raw_test_ece": raw_ece,
        "temp_test_ece": temp_ece,
        "temperature": fit.temperature,
    }
    with (reports_dir / "ece_sweep.json").open("w", encoding="utf-8") as f:
        json.dump(sweep_out, f, indent=2, sort_keys=True)

    from calrep.plotting import plot_ece_vs_bins

    bins = ece_bins
    raw_vals = [raw_ece[b] for b in bins]
    temp_vals = [temp_ece[b] for b in bins]
    plot_ece_vs_bins(
        bins,
        raw_vals,
        temp_vals,
        "ECE vs #bins (Test)",
        figures_dir / "ece_vs_bins.png",
    )

    print("\nECE vs bins (test)")
    for b in ece_bins:
        print(f"  bins={b:2d}: {raw_ece[b]:.4f} -> {temp_ece[b]:.4f}")

    temp_preds = test_logits_t.argmax(dim=1)
    acc_same = bool(torch.equal(raw_preds, temp_preds))

    # Compute metrics
    val_metrics = compute_metrics(
        val_logits_t, val_y, n_bins=n_bins, include_brier=True
    )
    test_metrics = compute_metrics(
        test_logits_t, test_y, n_bins=n_bins, include_brier=True
    )

    # Save temperature info
    temp_out = {
        "temperature": fit.temperature,
        "val_nll_before": fit.val_nll_before,
        "val_nll_after": fit.val_nll_after,
        "argmax_unchanged_on_test": acc_same,
    }
    with (reports_dir / "temperature.json").open("w", encoding="utf-8") as f:
        json.dump(temp_out, f, indent=2, sort_keys=True)

    # Save metrics
    metrics_out = {
        "n_bins": n_bins,
        "temperature": fit.temperature,
        "val": val_metrics,
        "test": test_metrics,
        "argmax_unchanged_on_test": acc_same,
    }
    with (reports_dir / "metrics_temp.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_out, f, indent=2, sort_keys=True)

    # Save reliability stats + plot
    test_rel = reliability_diagram_stats(test_logits_t, test_y, n_bins=n_bins)
    rel_out = {
        "n_bins": n_bins,
        "temperature": fit.temperature,
        "test": {
            "bin_edges": test_rel.bin_edges.tolist(),
            "bin_counts": test_rel.bin_counts.tolist(),
            "bin_acc": [None if x != x else float(x) for x in test_rel.bin_acc],
            "bin_conf": [None if x != x else float(x) for x in test_rel.bin_conf],
            "ece": test_rel.ece,
            "mce": test_rel.mce,
        },
    }
    with (reports_dir / "reliability_temp.json").open("w", encoding="utf-8") as f:
        json.dump(rel_out, f, indent=2, sort_keys=True)

    edges = np.array(rel_out["test"]["bin_edges"], dtype=float)
    acc = np.array(
        [np.nan if v is None else float(v) for v in rel_out["test"]["bin_acc"]],
        dtype=float,
    )
    conf = np.array(
        [np.nan if v is None else float(v) for v in rel_out["test"]["bin_conf"]],
        dtype=float,
    )

    out_fig = figures_dir / "reliability_temp.png"
    plot_reliability(
        edges,
        acc,
        conf,
        title="Reliability Diagram (Temperature Scaled, Test)",
        out_path=out_fig,
    )

    raw = json.load(open(reports_dir / "metrics_raw.json", "r", encoding="utf-8"))

    print("Temperature:", fit.temperature)
    print("Argmax unchanged on test:", acc_same)

    print("\nVAL  (raw -> temp)")
    print("  NLL:", raw["val"]["nll"], "->", val_metrics["nll"])
    print("  ECE:", raw["val"]["ece"], "->", val_metrics["ece"])

    print("\nTEST (raw -> temp)")
    print("  Acc:", raw["test"]["accuracy"], "->", test_metrics["accuracy"])
    print("  NLL:", raw["test"]["nll"], "->", test_metrics["nll"])
    print("  ECE:", raw["test"]["ece"], "->", test_metrics["ece"])
    print("  MCE:", raw["test"]["mce"], "->", test_metrics["mce"])


if __name__ == "__main__":
    main()
