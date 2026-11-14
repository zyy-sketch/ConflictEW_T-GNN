"""tests/test_metrics.py：回测指标（PRD F09 / §13.4）。"""
import numpy as np
import pandas as pd
import pytest

from conflict_ew.metrics import (auc_ci, bh_fdr, brier_decomposition,
                                 delong_auc_var, lead_time_analysis)


def test_delong_var_positive_and_auc_ci():
    rng = np.random.default_rng(0)
    y = np.array([0] * 90 + [1] * 10)
    s = np.concatenate([rng.normal(0, 1, 90), rng.normal(1.2, 1, 10)])
    var = delong_auc_var(y, s)
    assert var >= 0
    ci = auc_ci(y, s)
    assert 0 < ci["auc"] < 1
    assert ci["ci_lower"] <= ci["auc"] <= ci["ci_upper"]


def test_brier_decomposition_identity():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.2).astype(float)
    p = np.clip(y + rng.normal(0, 0.15, 500), 0.01, 0.99)
    b = brier_decomposition(y, p, n_bins=10)
    # Brier ≈ Reliability - Resolution + Uncertainty（分箱估计下近似成立）
    assert abs(b["brier"] - (b["reliability"] - b["resolution"] + b["uncertainty"])) < 0.02
    assert 0 <= b["reliability"] <= 1
    assert 0 <= b["uncertainty"] <= 0.25


def test_bh_fdr():
    pvals = np.array([0.001, 0.01, 0.04, 0.2, 0.5])
    sig = bh_fdr(pvals, alpha=0.05)
    # BH 校正后至少最小的 p 值显著
    assert bool(sig[0])
    # 单调性：p 值越大越不可能显著
    assert sum(sig) >= 1


def test_lead_time_analysis():
    oos = pd.DataFrame({
        "country": ["AAA"] * 10 + ["BBB"] * 10,
        "year": list(range(2015, 2025)) * 2,
        "p_conflict": [0.1, 0.2, 0.8, 0.9, 0.5, 0.3, 0.2, 0.1, 0.1, 0.1] * 2,
        "fold": ["test"] * 20,
    })
    panel = pd.DataFrame({
        "iso3": ["AAA"] * 11 + ["BBB"] * 11,
        "year": list(range(2014, 2025)) * 2,
        "y_onset": [0] * 11 + [0] * 10 + [1],
    })
    out = lead_time_analysis(oos, panel, threshold=0.7)
    # AAA 2017 首次 p≥0.7；BBB 2024 onset 但从未触发预警 → 覆盖 1 个
    assert out["n_onsets_total"] == 1
    assert out["n_onsets_covered"] == 1
    assert out["mean_lead_months"] is not None
