"""Post-hoc calibration: temperature scaling + split conformal.

Temperature fixes systematic over/under-confidence. Split conformal
adds a distribution-free guarantee: with probability >= 1-alpha the
true label is inside the returned prediction set — a stronger claim
than empirical calibration, and one Jev does not make.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def fit_temperature(logits_list, labels):
    """Single scalar T minimizing NLL on held-out data.

    Optimized in log space so T stays positive. Logits are [K]
    tensors; labels are scalar class indices.
    """
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
    return float(log_T.detach().exp())


def conformal_threshold(probs_list, labels, alpha=0.1):
    """Split-conformal quantile of nonconformity 1 - p[true].

    Returns q: prediction sets {k : p_k >= 1 - q} have >= 1-alpha
    marginal coverage. Rank is ceil((n+1)(1-alpha)); beyond n the
    guarantee requires the full set (q = inf).
    """
    import math

    n = len(probs_list)
    assert n > 0 and 0 < alpha < 1
    scores = sorted(1.0 - float(p[y]) for p, y in zip(probs_list, labels))
    rank = math.ceil((n + 1) * (1 - alpha))
    return math.inf if rank > n else scores[rank - 1]


def prediction_set(probs, q):
    return [i for i, p in enumerate(probs) if float(p) >= 1.0 - q]


def binomial_upper_bound(errors, n, confidence=0.95):
    """One-sided Clopper-Pearson upper bound on the error rate.

    Certifies P(wrong | accepted) <= U at the given confidence, over
    n independent accepted examples. With zero errors this is
    1 - (1-confidence)**(1/n): ~299 accepted for U < 1%.
    """
    from scipy.stats import beta

    if n == 0:
        return 1.0
    assert 0 <= errors <= n
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
