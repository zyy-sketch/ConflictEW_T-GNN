"""Unit tests for evaluation metrics (SCI_OPTIMIZATION_PLAN §E3)."""
import numpy as np

from conflict_ew.eval import metrics


def test_auc_perfect_and_random():
    rng = np.random.default_rng(0)
    y = np.array([0, 0, 1, 1])
    assert metrics._auc_mannwhitney(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    yb = rng.integers(0, 2, 200)
    sb = rng.random(200)
    auc = metrics._auc_mannwhitney(yb, sb)
    assert 0.3 < auc < 0.7


def test_delong_ci_covers_true_value():
    rng = np.random.default_rng(1)
    n_pos, n_neg = 60, 60
    pos = rng.normal(1.0, 1.0, n_pos)
    neg = rng.normal(0.0, 1.0, n_neg)
    y = np.array([1] * n_pos + [0] * n_neg)
    s = np.concatenate([pos, neg])
    auc, (lo, hi) = metrics.auc_delong_ci(y, s)
    true = metrics._auc_mannwhitney(y, s)
    assert abs(auc - true) < 1e-9
    assert lo < auc < hi
    assert 0.6 < auc < 0.85


def test_brier_decomposition_sums():
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, 300)
    p = np.clip(rng.random(300), 0.01, 0.99)
    rel, res, unc = metrics.brier_decomposition(y, p, n_bins=10)
    # Bin-mean Murphy decomposition: equals raw Brier up to a within-bin
    # heterogeneity term, which is small with 10 bins.
    assert abs(rel - res + unc - metrics.brier(y, p)) < 0.01
    # all three components are non-negative by construction
    assert rel >= 0 and res >= 0 and unc >= 0


def test_wilson_ci_known_values():
    # k=0, n=10 -> CI should include 0 and be finite
    lo, hi = metrics.wilson_ci(0, 10)
    assert lo == 0.0 and 0 < hi < 1
    lo, hi = metrics.wilson_ci(10, 10)
    assert 0 < lo < 1 and np.isclose(hi, 1.0)


def test_bootstrap_ci_contains_mean():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 100)
    p = np.clip(rng.random(100), 0.05, 0.95)
    mean, (lo, hi) = metrics.brier_bootstrap_ci(y, p, n_boot=200, seed=0)
    assert lo <= mean <= hi


def test_delong_test_same_scores_insignificant():
    """Identical scores -> zero diff, p close to 1."""
    rng = np.random.default_rng(4)
    y = np.array([1] * 50 + [0] * 50)
    s = rng.normal(0, 1, 100)
    res = metrics.delong_test(y, s, s.copy())
    assert abs(res["diff"]) < 1e-9
    assert res["p_value"] > 0.5


def test_delong_test_separable_significant():
    """Clearly separable scores -> significant difference."""
    rng = np.random.default_rng(5)
    n_pos, n_neg = 80, 80
    y = np.array([1] * n_pos + [0] * n_neg)
    good = np.concatenate([rng.normal(2.0, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)])
    bad = np.concatenate([rng.normal(0.0, 1.0, n_pos), rng.normal(0.0, 1.0, n_neg)])
    res = metrics.delong_test(y, good, bad)
    assert res["auc_a"] > res["auc_b"]
    assert res["p_value"] < 0.05


def test_benjamini_hochberg_known_values():
    """BH step-down matches the textbook example (Benjamini & Hochberg 1995)."""
    p = np.array([0.01, 0.04, 0.03, 0.20])
    adj, rej = metrics.benjamini_hochberg(p, alpha=0.05)
    # smallest p (0.01) is the only rejection at alpha=0.05 (m=4)
    assert rej[0]
    assert not rej[1] and not rej[2] and not rej[3]
    # adjusted p-values are non-decreasing with the original p order
    order = np.argsort(p)
    assert np.all(np.diff(adj[order]) >= 0)
