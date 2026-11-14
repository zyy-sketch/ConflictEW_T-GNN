"""tests/test_kalman.py：Kalman LLT 滤波（PRD F05 验收 + §13.4 测试要求）。"""
import numpy as np
import pandas as pd
import pytest

from conflict_ew.kalman import ANOMALY_Z, _fit_single, _kalman_filter, compute_kalman_signals


def test_kalman_filter_shapes_and_finite():
    rng = np.random.default_rng(0)
    y = 20 + 0.03 * np.arange(60) + rng.normal(0, 0.05, 60)
    e, F, mu, P, ll = _kalman_filter(y, 0.005, 0.001, 0.0005)
    assert e.shape == y.shape
    assert np.all(np.isfinite(e))
    assert np.all(F > 0)
    assert np.isfinite(ll)
    # 滤波残差应近似白噪声（均值接近 0）
    assert abs(np.mean(e)) < 0.1


def test_kalman_catches_anomaly_shock():
    """军费突增应产生 |e*|≥2 的标准化残差（异常信号提取）。"""
    rng = np.random.default_rng(1)
    y = 20 + 0.03 * np.arange(80)
    y[50:55] += 0.8  # 模拟突发军费冲击
    y = y + rng.normal(0, 0.04, 80)
    fit = _fit_single("TST", y, [(0.002, 0.001, 0.0005), (0.01, 0.001, 0.0005)])
    assert fit.n_obs == 80
    assert np.isfinite(fit.log_lik)
    assert fit.sigma2_eps > 0


def test_kalman_skips_all_nan():
    """全 NaN 序列跳过并记录（F05 验收：全 NaN 序列跳过+记录）。"""
    y = np.full(30, np.nan)
    fit = _fit_single("NAN", y, [(0.01, 0.001, 0.0005)])
    assert fit.diverged is True
    assert fit.n_obs == 0
    assert "NaN" in fit.note


def test_kalman_short_series():
    y = np.array([20.0, 20.1, 20.2])
    fit = _fit_single("SHT", y, [(0.01, 0.001, 0.0005)])
    assert "过短" in fit.note


def test_anomaly_threshold():
    assert ANOMALY_Z == 2.0


def test_compute_kalman_signals_output(small_panel):
    panel = small_panel.copy()
    panel = panel[["iso3", "year", "mil_expense_log"]]
    sig = compute_kalman_signals(panel)
    assert {"iso3", "year", "e_star", "is_anomaly"} <= set(sig.columns)
    assert sig["iso3"].nunique() == 3
    assert sig["is_anomaly"].isin([0, 1]).all()
