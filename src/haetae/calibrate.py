"""Post-hoc calibration: temperature scaling + split conformal.

Temperature fixes systematic over/under-confidence. Split conformal
adds a distribution-free guarantee: with probability >= 1-alpha the
true label is inside the returned prediction set — a stronger claim
than empirical calibration, and one Jev does not make.
"""

from __future__ import annotations

import math
import numbers
from pathlib import Path
import torch
import torch.nn.functional as F


def fit_temperature(logits_list, labels):
    """Single scalar T minimizing NLL on held-out data.

    Optimized in log space so T stays positive. Logits are [K]
    tensors; labels are scalar class indices.
    """
    if not logits_list or len(logits_list) != len(labels):
        raise ValueError("logits and labels must have equal nonzero length")
    for logits, label in zip(logits_list, labels):
        if logits.ndim != 1 or logits.numel() < 2:
            raise ValueError("each logit tensor must be 1D with >=2 classes")
        if not torch.isfinite(logits).all():
            raise ValueError("logits must be finite")
        if not isinstance(label, numbers.Integral) or not 0 <= label < logits.numel():
            raise ValueError("label is outside its logit tensor")

    log_T = torch.tensor(0.0, requires_grad=True)
    opt = torch.optim.LBFGS([log_T], lr=0.05, max_iter=100)
    logits_list = [l.detach().float().cpu() for l in logits_list]
    labels = [int(y) for y in labels]

    def closure():
        opt.zero_grad()
        T = log_T.exp()
        loss = sum(F.cross_entropy((l / T).unsqueeze(0),
                                   torch.tensor([y]))
                   for l, y in zip(logits_list, labels)) / len(logits_list)
        loss.backward()
        return loss

    opt.step(closure)
    temperature = float(log_T.detach().exp())
    if not math.isfinite(temperature) or temperature <= 0:
        raise RuntimeError("temperature optimization produced an invalid value")
    return temperature


def save_temperature_artifact(output, temperature, run, manifest,
                              calibration_data, max_len):
    """Bind a scalar temperature to one completed generation and dataset."""
    from .checkpoint import atomic_json_save
    from .model import PACKING_POLICY

    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    current = manifest["current"]
    artifact = {
        "version": 1,
        "run_id": run["run_id"],
        "checkpoint_generation": current["generation"],
        "checkpoint_sha256": current["sha256"],
        "tokenizer_fingerprint": run["spec"]["tokenizer_fingerprint"],
        "inference_policy": {
            "packing": PACKING_POLICY,
            "max_len": max_len,
            "temperature_scaling": "scalar-v1",
        },
        "calibration_data": calibration_data,
        "temperature": float(temperature),
    }
    atomic_json_save(artifact, Path(output) / "calibration.json")
    return artifact


def conformal_threshold(probs_list, labels, alpha=0.1):
    """Split-conformal quantile of nonconformity 1 - p[true].

    Returns q: prediction sets {k : p_k >= 1 - q} have >= 1-alpha
    marginal coverage. Rank is ceil((n+1)(1-alpha)); beyond n the
    guarantee requires the full set (q = inf).
    """
    import math

    n = len(probs_list)
    if n == 0 or n != len(labels):
        raise ValueError("probabilities and labels must have equal nonzero length")
    if not 0 < alpha < 1:
        raise ValueError("alpha must be between 0 and 1")
    for probs, label in zip(probs_list, labels):
        if probs.ndim != 1 or probs.numel() < 2:
            raise ValueError("each probability tensor must be 1D with >=2 classes")
        if not torch.isfinite(probs).all() or (probs < 0).any():
            raise ValueError("probabilities must be finite and nonnegative")
        if abs(float(probs.sum()) - 1.0) > 1e-3:
            raise ValueError("probabilities must sum to 1")
        if not isinstance(label, numbers.Integral) or not 0 <= label < probs.numel():
            raise ValueError("label is outside its probability tensor")
    scores = sorted(1.0 - float(p[y]) for p, y in zip(probs_list, labels))
    rank = math.ceil((n + 1) * (1 - alpha))
    return math.inf if rank > n else scores[rank - 1]


def prediction_set(probs, q):
    # Use the same score expression that produced q. Reconstructing a
    # probability cutoff as 1 - q can round upward at the inclusive boundary.
    return [i for i, p in enumerate(probs) if 1.0 - float(p) <= q]


def binomial_upper_bound(errors, n, confidence=0.95):
    """One-sided Clopper-Pearson upper bound on the error rate.

    Certifies P(wrong | accepted) <= U at the given confidence, over
    n independent accepted examples. With zero errors this is
    1 - (1-confidence)**(1/n): ~299 accepted for U < 1%.
    """
    from scipy.stats import beta

    if not isinstance(errors, int) or not isinstance(n, int):
        raise TypeError("errors and n must be integers")
    if n < 0 or not 0 <= errors <= n:
        raise ValueError("require 0 <= errors <= n")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")
    if n == 0:
        return 1.0
    if errors == 0:
        return 1.0 - (1.0 - confidence) ** (1.0 / n)
    if errors == n:
        return 1.0
    return float(beta.ppf(confidence, errors + 1, n - errors))


def certify_selective_risk(probs_list, labels, threshold, confidence=0.95):
    """Certify an act/abstain rule 'act when max prob >= threshold'.

    Returns accepted count, observed errors, and the certified upper
    bound on P(wrong | accepted). The rule must be frozen before the
    certification data is touched.
    """
    if not probs_list or len(probs_list) != len(labels):
        raise ValueError("probabilities and labels must have equal nonzero length")
    if not 0 <= threshold <= 1:
        raise ValueError("threshold must be between 0 and 1")
    # Reuse the strict distribution and label validation.
    conformal_threshold(probs_list, labels, alpha=0.5)
    accepted = [(p, y) for p, y in zip(probs_list, labels)
                if float(p.max()) >= threshold]
    errors = sum(1 for p, y in accepted if int(p.argmax()) != y)
    return {
        "threshold": threshold,
        "accepted": len(accepted),
        "coverage": len(accepted) / max(1, len(probs_list)),
        "errors": errors,
        "risk_upper_bound": binomial_upper_bound(errors, len(accepted), confidence),
    }
