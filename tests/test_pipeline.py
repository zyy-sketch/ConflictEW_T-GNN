"""End-to-end integration test on synthetic small data (SCI_OPTIMIZATION_PLAN §E4/E5)."""
import numpy as np
import pandas as pd
import pytest

from conflict_ew.data import build_panel
from conflict_ew.features.factors import build_features
from conflict_ew.features.kalman import fit_kalman
from conflict_ew.models.xgb import make_folds, prior_correct_probs, run_all_models


def _synthetic_raw(n_countries=6, n_years=30, seed=0):
    rng = np.random.default_rng(seed)
    codes = [f"C{i}" for i in range(n_countries)]
    spending = {}
    conflicts = []
    relations = []
    for code in codes:
        t = np.arange(n_years)
        base = rng.uniform(100, 500)
        trend = rng.uniform(-0.01, 0.03)
        vals = base * np.exp(trend * t) + rng.normal(0, base * 0.03, n_years)
        spending[code] = pd.DataFrame(
            {"year": 1990 + t, "value": np.maximum(vals, 1.0)}
        )
    for i in range(n_countries - 1):
        relations.append({"source": codes[i], "target": codes[i + 1], "type": "adjacency", "weight": 0.8})
    # a few conflicts, including one in the out-of-sample window (>= 2010)
    conflicts = [
        {"country": codes[1], "year": 2000, "event": "e", "type": "interstate", "severity": 3},
        {"country": codes[3], "year": 2005, "event": "e", "type": "civil", "severity": 4},
        {"country": codes[2], "year": 2012, "event": "e", "type": "interstate", "severity": 3},
    ]
    countries = pd.DataFrame(
        [{"id": c, "name": c, "region": "r", "lat": 0, "lng": 0, "population": 10, "gdp": 100} for c in codes]
    )
    return {
        "countries": countries,
        "spending": spending,
        "relations": pd.DataFrame(relations),
        "conflicts": pd.DataFrame(conflicts),
        "factor_scores": {},
        "alerts": [],
        "settings": {},
        "tgnn": {},
    }


def _cfg():
    return {
        "kalman": {
            "eps": 1e-6,
            "init_p": 1e6,
            "anomaly_sigma": 2.0,
            "grid_q_eta": [1e-4, 1e-3, 1e-2],
            "grid_q_zeta": [1e-7, 1e-5, 1e-3],
            "grid_r_eps": [1e-3, 1e-2, 1e-1],
        },
        "model": {
            "test_start": 2010,
            "n_splits": 2,
            "window_years": 2,
            "params": {"max_depth": 2, "learning_rate": 0.1, "n_estimators": 30},
            "rare_event": {"enabled": True, "pos_weight": 5.0, "prior_correction": True},
            "features": ["e_star", "log_level", "growth", "volatility", "slope", "neighbor_e"],
        },
    }


def test_pipeline_small():
    raw = _synthetic_raw()
    panel = build_panel(raw)
    panel, kalman_params, signals = build_features(raw, panel, _cfg())
    assert set(["e_star", "log_level", "growth", "volatility", "slope", "neighbor_e"]).issubset(panel.columns)
    assert len(kalman_params) == 6
    assert panel["label"].sum() >= 1

    cfg = _cfg()
    oos, folds = run_all_models(panel, cfg, seed=42)
    for name, df in oos.items():
        assert set(df.columns) == {"country", "year", "y_true", "y_prob", "fold"}
        assert df["y_prob"].between(0, 1).all(), name
    assert len(folds) == 2


def test_folds_are_blocked_and_expanding():
    panel = pd.DataFrame({"year": list(range(2000, 2021))})
    cfg = {"model": {"test_start": 2015, "n_splits": 3, "window_years": 2}}
    folds = make_folds(panel, cfg)
    assert len(folds) == 3
    train_sizes = [len(tr) for tr, _ in folds]
    assert train_sizes == sorted(train_sizes), "train windows must expand"
    # no overlap between train and test
    for tr, te in folds:
        assert set(tr).isdisjoint(set(te))


def test_kalman_no_nan_on_edge_cases():
    for y in ([0.0, 0.0, 0.0], [100.0], [100.0, 100.0, 1e6, 100.0]):
        res, params = fit_kalman(y, [1e-3], [1e-5], [1e-2])
        assert np.all(np.isfinite(res["e"]))


def test_prior_correct_probs_shrinks_rare_event():
    """When the true event rate tau is lower than the training rate, prior
    correction must shrink the predicted probabilities (King-Zeng 2001)."""
    y_train = np.array([0] * 20 + [1] * 20)  # training rate 0.5
    p = np.array([0.8, 0.5, 0.2])
    p_star = prior_correct_probs(p, y_train, tau=0.1)
    # tau=0.1 < ybar=0.5 -> c > 1 -> p* < p for all interior p
    assert np.all(p_star < p)
    assert np.all(p_star >= 0) and np.all(p_star <= 1)
    # no-op when tau equals the training rate
    np.testing.assert_allclose(prior_correct_probs(p, y_train, tau=0.5), p)


def test_prior_correct_probs_edge_rates():
    """Degenerate rates must not produce NaN."""
    p = np.array([0.5, 0.5])
    np.testing.assert_array_equal(prior_correct_probs(p, np.array([0, 0]), 0.1), p)
    np.testing.assert_array_equal(prior_correct_probs(p, np.array([1, 1]), 0.1), p)
    np.testing.assert_array_equal(prior_correct_probs(p, np.array([0, 1]), 0.0), p)
