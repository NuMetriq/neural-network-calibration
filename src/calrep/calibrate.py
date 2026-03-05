from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TemperatureFitResult:
    temperature: float
    val_nll_before: float
    val_nll_after: float
    steps: int


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    return logits / float(temperature)


def fit_temperature(
    val_logits: torch.Tensor,
    val_labels: torch.Tensor,
    max_iter: int = 200,
    lr: float = 0.05,
) -> TemperatureFitResult:
    """
    Fits a single scalar temperature T>0 by minimizing validation NLL of softmax(logits/T).
    Optimize log(T) for positivity.
    """
    device = val_logits.device
    # Start near 1.0
    logT = torch.zeros((), device=device, requires_grad=True)

    optimizer = torch.optim.LBFGS(
        [logT], lr=lr, max_iter=max_iter, line_search_fn="strong_wolfe"
    )

    nll_before = float(F.cross_entropy(val_logits, val_labels).item())

    def closure():
        optimizer.zero_grad(set_to_none=True)
        T = torch.exp(logT)
        scaled = val_logits / T
        loss = F.cross_entropy(scaled, val_labels)
        loss.backward()
        return loss

    loss = optimizer.step(closure)
    T_star = float(torch.exp(logT).detach().item())

    nll_after = float(F.cross_entropy(val_logits / T_star, val_labels).item())

    return TemperatureFitResult(
        temperature=T_star,
        val_nll_before=nll_before,
        val_nll_after=nll_after,
        steps=max_iter,  # LBFGS internal; keep simple for logging
    )
