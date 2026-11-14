"""tests/test_factors.py：6 因子构造（PRD F06 / §13.4 新增）。"""
import numpy as np
import pandas as pd
import pytest

from conflict_ew.factors import (FACTOR_DEFINITIONS, build_features,
                                 factor_column_map, get_factor_columns)


@pytest.fixture
def panel_with_signal():
    rng = np.random.default_rng(3)
    years = list(range(1990, 2015))
    rows = []
    for iso in ("AAA", "BBB", "CCC"):
        for y in years:
            rows.append({
                "iso3": iso, "year": y,
                "mil_expense_log": 20 + 0.02 * (y - 1990) + rng.normal(0, 0.05),
                "gdp_growth": float(rng.normal(3, 1)),
                "inflation": float(abs(rng.normal(4, 1))),
                "unemployment": float(rng.normal(7, 1)),
                "polity_score": int(rng.integers(-3, 4)),
                "vdem_ldi": float(rng.uniform(0.2, 0.8)),
                "icow_claim": int(rng.random() < 0.2),
                "atop_change": int(rng.random() < 0.15),
                "y_onset": 0,
            })
    return pd.DataFrame(rows)


@pytest.fixture
def signals():
    rng = np.random.default_rng(5)
    years = list(range(1990, 2015))
    rows = []
    for iso in ("AAA", "BBB", "CCC"):
        for y in years:
            e = 2.5 if (iso == "AAA" and 2002 <= y <= 2004) else float(rng.normal(0, 0.8))
            rows.append({"iso3": iso, "year": y, "e_star": e,
                         "is_anomaly": int(abs(e) >= 2)})
    return pd.DataFrame(rows)


def test_factor_definitions_complete():
    ids = {f["id"] for f in FACTOR_DEFINITIONS}
    assert ids == {"F1", "F2", "F3", "F4", "F5", "F6"}


def test_factor_columns_registry():
    cols = get_factor_columns()
    assert len(cols) == 9  # F1×1 + F2×1 + F3×3 + F4×2 + F5×1 + F6×1
    m = factor_column_map()
    assert set(m.values()) == {"F1", "F2", "F3", "F4", "F5", "F6"}


def test_build_features_all_factors(panel_with_signal, signals):
    feats = build_features(panel_with_signal, signals, None, out_path=None)
    feats = feats.set_index(["iso3", "year"])
    assert feats.shape[0] == 3 * 25
    for c in get_factor_columns():
        assert c in feats.columns
    # F1 异常信号被正确带入（AAA 2002-2004 e*=2.5）
    assert feats.loc[("AAA", 2003), "F1_e_star"] == pytest.approx(2.5)
    # F2 无邻接矩阵时降级（degraded=1）
    assert feats.loc[("AAA", 2003), "F2_degraded"] == 1
    # 经济因子滞后 1 年（F3_*_lag1）
    assert "F3_gdp_growth_lag1" in feats.columns


def test_build_features_with_adjacency(panel_with_signal, signals):
    adj = pd.DataFrame({"iso3_a": ["AAA", "BBB"], "iso3_b": ["BBB", "CCC"]})
    feats = build_features(panel_with_signal, signals, adj, out_path=None)
    feats = feats.set_index(["iso3", "year"])
    # 有邻接时 F2 不降级
    assert int(feats.loc[("AAA", 2003), "F2_degraded"]) == 0
    assert np.isfinite(feats.loc[("AAA", 2003), "F2_arms_race"])
