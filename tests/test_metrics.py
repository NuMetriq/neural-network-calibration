import torch
import torch.nn.functional as F
from calrep.calibrate import apply_temperature
from calrep.metrics import (
    adaptive_ece,
    compute_metrics,
    nll_from_logits,
    reliability_diagram_stats,
)


def test_nll_matches_torch_cross_entropy():
    torch.manual_seed(0)
    logits = torch.randn(128, 10)
    y = torch.randint(0, 10, (128,))
    a = nll_from_logits(logits, y)
    b = float(F.cross_entropy(logits, y).item())
    assert abs(a - b) < 1e-12


def test_ace_near_zero_for_perfect_calibration_binary():
    torch.manual_seed(0)
    n = 20000
    p = torch.empty(n)
    p[: n // 2] = 0.8
    p[n // 2 :] = 0.2
    y = torch.bernoulli(p).long()

    logit = torch.log(p / (1.0 - p))
    logits = torch.stack([torch.zeros(n), logit], dim=1)

    ace = adaptive_ece(logits, y, n_bins=15)
    assert ace < 0.01


def test_ace_large_for_confidently_wrong_predictions():
    n = 5000
    logits = torch.tensor([[10.0, -10.0]]).repeat(n, 1)
    y = torch.ones(n, dtype=torch.long)

    ace = adaptive_ece(logits, y, n_bins=15)
    assert ace > 0.9


def test_reliability_bin_counts_sum_to_n():
    torch.manual_seed(0)
    logits = torch.randn(1000, 5)
    y = torch.randint(0, 5, (1000,))
    rel = reliability_diagram_stats(logits, y, n_bins=15)
    assert int(rel.bin_counts.sum()) == 1000


def test_temperature_scaling_does_not_change_argmax():
    torch.manual_seed(0)
    logits = torch.randn(256, 7)
    preds_raw = logits.argmax(dim=1)

    scaled = apply_temperature(logits, temperature=3.7)
    preds_scaled = scaled.argmax(dim=1)

    assert torch.equal(preds_raw, preds_scaled)


def test_temperature_scaling_handles_confidence_one_edge_case():
    """
    This guards against a common ECE binning bug: confidences exactly equal to 1.0
    can fall outside the last bin depending on binning logic.

    We construct logits that produce softmax confidence extremely close to 1.0 for class 0,
    and ensure reliability stats still count every sample.
    """
    n = 1000
    logits = torch.tensor([[1000.0, -1000.0]]).repeat(n, 1)  # softmax ~ [1.0, 0.0]
    y = torch.zeros(n, dtype=torch.long)

    rel = reliability_diagram_stats(logits, y, n_bins=15)
    assert int(rel.bin_counts.sum()) == n
    assert rel.ece >= 0.0
    assert rel.mce >= 0.0


def test_ece_near_zero_for_perfect_calibration_binary():
    """
    Construct a perfectly calibrated binary classifier:
    - For each sample i, predicted confidence p_i is either 0.2 or 0.8.
    - Labels are generated so that P(y=1|p=0.8)=0.8 and P(y=1|p=0.2)=0.2.
    Then in expectation, accuracy within each confidence bin matches confidence.
    With enough samples, ECE should be small.
    """
    torch.manual_seed(0)
    n = 20000

    # Half with p=0.8, half with p=0.2
    p_high = 0.8
    p_low = 0.2
    p = torch.empty(n)
    p[: n // 2] = p_high
    p[n // 2 :] = p_low

    # Generate labels with matching probabilities
    y = torch.bernoulli(p).long()  # y in {0,1}

    # Create logits that correspond to these probabilities exactly:
    # For binary with logits [0, logit(p/(1-p))], softmax gives probs [1-p, p]
    logit = torch.log(p / (1.0 - p))
    logits = torch.stack([torch.zeros(n), logit], dim=1)

    rel = reliability_diagram_stats(logits, y, n_bins=15)
    assert rel.ece < 0.01  # should be quite small with n=20k


def test_ece_large_for_confidently_wrong_predictions():
    """
    Build a classifier that is confidently wrong:
    - Always predicts class 0 with confidence ~1.0
    - True labels are always class 1
    Expect ECE to be near 1 (or at least very large).
    """
    n = 5000
    # 2-class logits: make class 0 extremely large
    logits = torch.tensor([[10.0, -10.0]]).repeat(n, 1)
    y = torch.ones(n, dtype=torch.long)  # always class 1

    rel = reliability_diagram_stats(logits, y, n_bins=15)
    assert rel.ece > 0.9
    assert rel.mce > 0.9


def test_compute_metrics_contains_expected_keys():
    torch.manual_seed(0)
    logits = torch.randn(512, 3)
    y = torch.randint(0, 3, (512,))
    m = compute_metrics(logits, y, n_bins=15, include_brier=True)

    for k in ["accuracy", "nll", "ece", "mce", "brier", "ece_bins"]:
        assert k in m
