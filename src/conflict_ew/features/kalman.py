"""Kalman-filter local-linear-trend (LLT) signal extraction.

Mirrors ``kalmanLLT`` in ``dashboard_v3.html`` so the Python pipeline and the
browser prototype agree numerically. State is ``[mu (level), beta (slope)]``;
the observation model measures ``mu`` with Gaussian noise of variance ``r_eps``.
The standardized one-step-ahead prediction residual ``e* = v / sqrt(S)`` is the
core conflict-risk signal used by the model.
"""
from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np


def kalman_llt(
    y: Sequence[float],
    q_eta: float,
    q_zeta: float,
    r_eps: float,
    init_p: float = 1e6,
) -> dict[str, Any]:
    """Run the LLT Kalman filter over a univariate series.

    Parameters
    ----------
    y : array_like
        Log-transformed military expenditure series.
    q_eta, q_zeta, r_eps : float
        Process variances (level, slope) and observation variance.
    init_p : float
        Diffuse initial state variance.

    Returns
    -------
    dict
        ``pred`` (one-step prediction), ``v`` (innovation), ``S`` (innovation
        variance), ``e`` (standardized residual e*), ``mu``, ``beta``, ``ll``
        (log-likelihood via prediction-error decomposition).
    """
    y = np.asarray(y, dtype=float)
    n = y.shape[0]
    mu = float(y[0])
    beta = 0.0
    P11 = float(init_p)
    P22 = float(init_p)
    P12 = 0.0

    pred = np.empty(n)
    v = np.empty(n)
    S = np.empty(n)
    e = np.empty(n)
    mu_arr = np.empty(n)
    beta_arr = np.empty(n)
    ll = 0.0

    for t in range(n):
        mup = mu + beta
        bp = beta
        P11p = P11 + 2.0 * P12 + P22 + q_eta
        P12p = P12 + P22
        P22p = P22 + q_zeta
        vt = y[t] - mup
        St = P11p + r_eps
        K1 = P11p / St
        K2 = P12p / St
        mu = mup + K1 * vt
        beta = bp + K2 * vt
        P11u = (1.0 - K1) * P11p
        P12u = (1.0 - K1) * P12p
        P22u = P22p - K2 * P12p
        P11, P12, P22 = P11u, P12u, P22u
        ll += -0.5 * (math.log(2.0 * math.pi * St) + vt * vt / St)
        pred[t] = mup
        v[t] = vt
        S[t] = St
        e[t] = vt / math.sqrt(St)
        mu_arr[t] = mu
        beta_arr[t] = beta

    return {"pred": pred, "v": v, "S": S, "e": e, "mu": mu_arr, "beta": beta_arr, "ll": ll}


def fit_kalman(
    y: Sequence[float],
    grid_q_eta: Sequence[float],
    grid_q_zeta: Sequence[float],
    grid_r_eps: Sequence[float],
    init_p: float = 1e6,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Maximum-likelihood fit via prediction-error decomposition (grid search).

    Returns
    -------
    (result, params)
        ``result`` is the filtered series; ``params`` records the fitted
        ``{q_eta, q_zeta, r_eps, ll, n}`` for provenance (SCI_OPTIMIZATION_PLAN §A3).
    """
    best: dict[str, Any] | None = None
    best_ll = -np.inf
    best_params: dict[str, Any] | None = None
    for qe in grid_q_eta:
        for qz in grid_q_zeta:
            for re_ in grid_r_eps:
                r = kalman_llt(y, qe, qz, re_, init_p)
                if r["ll"] > best_ll:
                    best_ll = r["ll"]
                    best = r
                    best_params = {
                        "q_eta": qe,
                        "q_zeta": qz,
                        "r_eps": re_,
                        "ll": r["ll"],
                        "n": len(y),
                    }
    assert best is not None
    best["params"] = best_params
    return best, best_params
