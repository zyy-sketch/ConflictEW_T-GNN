"""Unit tests for the Kalman local-linear-trend filter (SCI_OPTIMIZATION_PLAN §E1)."""
import numpy as np

from conflict_ew.features.kalman import fit_kalman, kalman_llt

GRID_ETA = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2]
GRID_ZETA = [1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
GRID_EPS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1]


def test_e_star_standard_normal_under_null():
    """Under H0 (no anomaly), e* should be roughly standard normal."""
    rng = np.random.default_rng(0)
    # smooth trend + small noise
    n = 200
    t = np.arange(n)
    y = 2.0 + 0.01 * t + 0.05 * rng.standard_normal(n)
    res = kalman_llt(y, 1e-3, 1e-5, 1e-2)
    e = res["e"][20:]  # discard burn-in
    assert abs(e.mean()) < 0.3, f"mean {e.mean()}"
    # e* is a standardized innovation, hence order-1 variance (not exactly 1
    # because the test model's r_eps is slightly mis-specified vs the noise).
    assert 0.3 < e.std() < 1.5, f"std {e.std()}"


def test_recovers_positive_trend():
    """The slope should track a known positive linear trend."""
    rng = np.random.default_rng(1)
    n = 100
    t = np.arange(n)
    y = 1.0 + 0.05 * t + 0.01 * rng.standard_normal(n)
    res = kalman_llt(y, 1e-3, 1e-5, 1e-2)
    assert res["beta"][-1] > 0.03, f"final slope {res['beta'][-1]}"


def test_determinism():
    """Identical inputs produce identical outputs (no hidden randomness)."""
    rng = np.random.default_rng(2)
    y = 2.0 + 0.01 * np.arange(80) + 0.05 * rng.standard_normal(80)
    r1 = kalman_llt(y, 1e-3, 1e-5, 1e-2)
    r2 = kalman_llt(y, 1e-3, 1e-5, 1e-2)
    np.testing.assert_array_equal(r1["e"], r2["e"])
    assert r1["ll"] == r2["ll"]


def test_fit_returns_params_and_improves_ll():
    """MLE grid search should return params and a finite log-likelihood."""
    rng = np.random.default_rng(3)
    y = 3.0 + 0.02 * np.arange(60) + 0.1 * rng.standard_normal(60)
    res, params = fit_kalman(y, GRID_ETA, GRID_ZETA, GRID_EPS)
    assert set(params) >= {"q_eta", "q_zeta", "r_eps", "ll", "n"}
    assert np.isfinite(params["ll"])
    assert res["params"]["ll"] == params["ll"]


def test_log_transform_handles_zeros():
    """Log transform with floor should not produce NaN for zero/negative values."""
    y = np.log(np.maximum([0.0, -5.0, 10.0, 20.0], 1e-6))
    res = kalman_llt(y, 1e-3, 1e-5, 1e-2)
    assert np.all(np.isfinite(res["e"]))
