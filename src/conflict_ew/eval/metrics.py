"""Evaluation metrics with confidence intervals and significance tests.

Implements the statistical machinery promised by the paper (§4): DeLong CI for
AUC, bootstrap CI for Brier, Wilson CI for precision/recall, and a paired
bootstrap p-value for model comparisons.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats


def _auc_mannwhitney(y_true, y_score) -> float:
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = stats.rankdata(np.concatenate([pos, neg]))
    r_pos = ranks[: len(pos)]
    return float((r_pos.sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def auc(y_true, y_score) -> float:
    """Area under the ROC curve via the Mann-Whitney U statistic.

    Unlike ``auc_delong_ci`` (which needs ≥2 positives *and* ≥2 negatives to
    estimate a variance), this returns a valid point estimate whenever both
    classes are present — the right tool for per-fold AUC on rare events.
    """
    return _auc_mannwhitney(y_true, y_score)


def _placements(y_true, y_score) -> tuple[np.ndarray, np.ndarray, int, int]:
    """DeLong placement values for a single model.

    Returns ``(V10, V01, m, n)`` where ``V10`` (shape ``m``) is the placement
    of each positive observation against all negatives, and ``V01`` (shape
    ``n``) is the placement of each negative against all positives. Ties are
    counted as 0.5 (DeLong et al. 1988).
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    m, n = len(pos), len(neg)
    V10 = np.array([np.mean((pos[i] > neg) + 0.5 * (pos[i] == neg)) for i in range(m)])
    V01 = np.array([np.mean((pos > neg[j]) + 0.5 * (pos == neg[j])) for j in range(n)])
    return V10, V01, m, n


def _auc_from_placements(V10: np.ndarray) -> float:
    return float(V10.mean())


def auc_delong_ci(y_true, y_score, alpha: float = 0.05) -> tuple[float, tuple[float, float]]:
    """DeLong (1988) asymptotic CI for AUC. Returns ``(auc, (lo, hi))``."""
    V10, V01, m, n = _placements(y_true, y_score)
    if m < 2 or n < 2:
        return float("nan"), (float("nan"), float("nan"))
    auc = _auc_from_placements(V10)
    s10 = np.var(V10, ddof=1)
    s01 = np.var(V01, ddof=1)
    var = s10 / m + s01 / n
    se = float(np.sqrt(var))
    z = float(stats.norm.ppf(1 - alpha / 2))
    return auc, (auc - z * se, auc + z * se)


def delong_test(
    y_true, score_a, score_b, alpha: float = 0.05
) -> dict[str, float]:
    """DeLong (1988) two-sided test for two correlated AUCs on the same data.

    Compares ``AUC(score_a)`` vs ``AUC(score_b)`` using the covariance of the
    placement values. Returns ``{auc_a, auc_b, diff, se_diff, z, p_value}``
    (SCI_OPTIMIZATION_PLAN.md §B1 — model vs baseline significance).
    """
    y_true = np.asarray(y_true)
    V10_a, V01_a, m, n = _placements(y_true, score_a)
    V10_b, V01_b, m_b, n_b = _placements(y_true, score_b)
    if m < 2 or n < 2 or m_b < 2 or n_b < 2 or m != m_b or n != n_b:
        return {
            "auc_a": float("nan"), "auc_b": float("nan"), "diff": float("nan"),
            "se_diff": float("nan"), "z": float("nan"), "p_value": float("nan"),
        }
    auc_a = _auc_from_placements(V10_a)
    auc_b = _auc_from_placements(V10_b)
    # DeLong covariance: cov(θ_a, θ_b) = s10_ab/m + s01_ab/n
    s10_ab = np.cov(V10_a, V10_b, ddof=1)[0, 1]
    s01_ab = np.cov(V01_a, V01_b, ddof=1)[0, 1]
    var_a = np.var(V10_a, ddof=1) / m + np.var(V01_a, ddof=1) / n
    var_b = np.var(V10_b, ddof=1) / m + np.var(V01_b, ddof=1) / n
    var_diff = var_a + var_b - 2.0 * (s10_ab / m + s01_ab / n)
    se_diff = float(np.sqrt(max(var_diff, 0.0)))
    diff = auc_a - auc_b
    if se_diff > 0.0:
        z = diff / se_diff
        p_value = float(2.0 * (1.0 - stats.norm.cdf(abs(z))))
    else:
        # identical models (or degenerate covariance): no evidence of difference
        z = 0.0 if abs(diff) < 1e-12 else float("inf")
        p_value = 1.0 if abs(diff) < 1e-12 else 0.0
    return {"auc_a": auc_a, "auc_b": auc_b, "diff": diff, "se_diff": se_diff, "z": z, "p_value": p_value}


def benjamini_hochberg(pvalues: Sequence[float], alpha: float = 0.05) -> tuple[np.ndarray, np.ndarray]:
    """Benjamini–Hochberg FDR control (SCI_OPTIMIZATION_PLAN.md §B2).

    Returns ``(adjusted_pvalues, rejected_mask)``. ``adjusted_pvalues`` are
    sorted in the *same order* as the input; ``rejected_mask`` flags hypotheses
    rejected at level ``alpha``.
    """
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adjusted = np.empty(m)
    rejected = np.zeros(m, dtype=bool)
    # step-down: iterate from largest to smallest p-value; adjusted values are
    # non-decreasing in the original p order (monotone via min with successor).
    prev = 1.0
    for rank in range(m, 0, -1):
        idx = order[rank - 1]
        bh = min(p[idx] * m / rank, 1.0)
        adjusted[idx] = min(prev, bh)
        prev = adjusted[idx]
    # rejection: largest k such that p_(k) <= alpha * k / m
    for rank in range(1, m + 1):
        idx = order[rank - 1]
        if p[idx] <= alpha * rank / m:
            rejected[idx] = True
        else:
            break
    return adjusted, rejected


def brier(y_true, y_prob) -> float:
    return float(np.mean((np.asarray(y_prob) - np.asarray(y_true)) ** 2))


def brier_decomposition(y_true, y_prob, n_bins: int = 10) -> tuple[float, float, float]:
    """Murphy (1973) Brier-score decomposition into Reliability, Resolution, Uncertainty.

    Uses bin-mean reliability ``(1/N) Σ_k n_k (p̄_k - ȳ_k)²`` (the quantity shown
    on a reliability diagram). For continuous predictions this reproduces the
    raw Brier score only up to a within-bin heterogeneity term, so the identity
    ``Brier = Reliability - Resolution + Uncertainty`` holds approximately (the
    residual is small with a reasonable number of bins).
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    N = len(y_true)
    o_bar = float(y_true.mean())
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    for k in range(n_bins):
        if k == n_bins - 1:
            mask = (y_prob >= bins[k]) & (y_prob <= bins[k + 1])
        else:
            mask = (y_prob >= bins[k]) & (y_prob < bins[k + 1])
        nk = int(mask.sum())
        if nk == 0:
            continue
        ok = float(y_true[mask].mean())
        pk = float(y_prob[mask].mean())
        reliability += nk * (pk - ok) ** 2 / N
        resolution += nk * (ok - o_bar) ** 2 / N
    uncertainty = o_bar * (1 - o_bar)
    return reliability, resolution, uncertainty


def brier_bootstrap_ci(
    y_true, y_prob, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, tuple[float, float]]:
    """Bootstrap (percentile) CI for Brier score. Returns ``(mean, (lo, hi))``."""
    rng = np.random.default_rng(seed)
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    idx = np.arange(len(y_true))
    bs = np.empty(n_boot)
    for b in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        bs[b] = brier(y_true[s], y_prob[s])
    lo = float(np.percentile(bs, 100 * alpha / 2))
    hi = float(np.percentile(bs, 100 * (1 - alpha / 2)))
    return float(bs.mean()), (lo, hi)


def wilson_ci(k, n, alpha: float = 0.05) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    z = stats.norm.ppf(1 - alpha / 2)
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def paired_bootstrap_pvalue(
    y_true, score_a, score_b, n_boot: int = 1000, seed: int = 0
) -> tuple[float, float]:
    """Two-sided paired-bootstrap p-value for ``AUC(score_a) > AUC(score_b)``."""
    rng = np.random.default_rng(seed)
    idx = np.arange(len(y_true))
    obs = _auc_mannwhitney(y_true, score_a) - _auc_mannwhitney(y_true, score_b)
    count = 0
    for _ in range(n_boot):
        s = rng.choice(idx, size=len(idx), replace=True)
        d = _auc_mannwhitney(y_true[s], score_a[s]) - _auc_mannwhitney(y_true[s], score_b[s])
        if d <= 0:
            count += 1
    p_one = count / n_boot
    return float(obs), 2 * min(p_one, 1 - p_one)


def precision_recall_at(y_true, y_prob, threshold: float) -> tuple[float, float, float]:
    pred = (np.asarray(y_prob) >= threshold).astype(int)
    tp = int(((pred == 1) & (np.asarray(y_true) == 1)).sum())
    fp = int(((pred == 1) & (np.asarray(y_true) == 0)).sum())
    fn = int(((pred == 0) & (np.asarray(y_true) == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall, 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
