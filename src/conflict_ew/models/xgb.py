"""Model training with blocked (expanding-window) temporal CV and rare-event weighting."""
from __future__ import annotations

import math
from typing import Any, Callable

import numpy as np
import pandas as pd

from ..features.factors import FEATURE_COLS


def prior_correct_probs(p: np.ndarray, y_train: np.ndarray, tau: float) -> np.ndarray:
    """King–Zeng (2001) prior correction for rare-event classification.

    Adjusts predicted probabilities so they reflect the true population event
    rate ``tau`` rather than the (over/under-sampled) training rate. This is the
    ``logit + prior correction`` strategy in SCI_OPTIMIZATION_PLAN.md §B5.

    Derivation: correct the intercept so ``logit(p*) = logit(p) - ln(c)`` with
    ``c = ((1-τ)/τ) · (ȳ/(1-ȳ))``, i.e. ``p* = p / (p + c·(1-p))``.
    """
    p = np.asarray(p, dtype=float)
    ybar = float(np.mean(np.asarray(y_train)))
    if not (0.0 < ybar < 1.0) or not (0.0 < tau < 1.0):
        return p
    c = ((1.0 - tau) / tau) * (ybar / (1.0 - ybar))
    return p / (p + c * (1.0 - p))


def make_folds(panel: pd.DataFrame, cfg: dict[str, Any]) -> list[tuple[list[int], list[int]]]:
    """Build expanding-window temporal folds.

    Test years are ``>= test_start``, grouped into windows of ``window_years``;
    each fold trains on all years strictly before its test window. This is a
    *blocked* split (no country-year leakage across folds).
    """
    test_start = int(cfg["model"]["test_start"])
    n_splits = int(cfg["model"]["n_splits"])
    window = int(cfg["model"]["window_years"])
    years = sorted(panel["year"].unique().tolist())
    test_years = [y for y in years if y >= test_start]
    folds: list[tuple[list[int], list[int]]] = []
    for k in range(n_splits):
        te = test_years[k * window : (k + 1) * window]
        if not te:
            break
        tr = [y for y in years if y < te[0]]
        folds.append((tr, te))
    return folds


def _make_xgb(cfg: dict[str, Any], seed: int, params_override: dict[str, Any] | None = None):
    import xgboost as xgb

    params = dict(cfg["model"]["params"])
    if params_override:
        params.update(params_override)
    pos_weight = float(cfg["model"]["rare_event"]["pos_weight"])
    return xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=pos_weight,
        random_state=seed,
        n_jobs=1,
        **params,
    )


def _make_logit(cfg: dict[str, Any], seed: int):
    from sklearn.linear_model import LogisticRegression

    # King-Zeng prior correction (applied post-hoc) replaces class weighting so
    # the two rare-event strategies — XGBoost scale_pos_weight vs logit prior
    # correction — are cleanly separated (SCI_OPTIMIZATION_PLAN.md §B5).
    prior = bool(cfg["model"]["rare_event"]["prior_correction"])
    return LogisticRegression(
        class_weight=None if prior else "balanced",
        max_iter=2000,
        random_state=seed,
    )


def _run_cv(
    panel: pd.DataFrame,
    folds: list[tuple[list[int], list[int]]],
    features: list[str],
    model_fn: Callable[[int], Any],
    seed: int,
    prior_tau: float | None = None,
) -> pd.DataFrame:
    """Run blocked CV, returning out-of-sample predictions as a DataFrame.

    If ``prior_tau`` is set, applies King-Zeng prior correction to the
    out-of-sample probabilities (for the logistic baseline).
    """
    oos_rows: list[dict[str, Any]] = []
    for fi, (tr, te) in enumerate(folds):
        trm = panel[panel["year"].isin(tr)]
        tem = panel[panel["year"].isin(te)].reset_index(drop=True)
        if trm.empty or tem.empty:
            continue
        Xtr = trm[features].astype(float).values
        ytr = trm["label"].astype(int).values
        Xte = tem[features].astype(float).values
        if ytr.sum() == 0 or ytr.sum() == len(ytr):
            p = np.full(len(tem), float(ytr.mean()))
        else:
            model = model_fn(seed)
            model.fit(Xtr, ytr)
            p = model.predict_proba(Xte)[:, 1]
            if prior_tau is not None:
                p = prior_correct_probs(p, ytr, prior_tau)
        for i, (_, row) in enumerate(tem.iterrows()):
            oos_rows.append(
                {
                    "country": row["country"],
                    "year": int(row["year"]),
                    "y_true": int(row["label"]),
                    "y_prob": float(p[i]),
                    "fold": fi,
                }
            )
    return pd.DataFrame(oos_rows)


def run_all_models(
    panel: pd.DataFrame,
    cfg: dict[str, Any],
    seed: int,
) -> dict[str, pd.DataFrame]:
    """Train all configured models + baselines; return name -> OOS predictions.

    Models:
      * ``xgb``            — full feature set (scale_pos_weight rare-event weighting)
      * ``kalman_only``    — only ``e_star`` (ablation: structural features removed)
      * ``logistic_rare_event`` — rare-event logistic baseline with King-Zeng prior correction
    """
    folds = make_folds(panel, cfg)
    features = list(cfg["model"]["features"])

    prior_tau: float | None = None
    rare = cfg["model"]["rare_event"]
    if rare.get("prior_correction"):
        prior_tau = float(rare.get("prior_tau", 0.5))

    out: dict[str, pd.DataFrame] = {}
    out["xgb"] = _run_cv(panel, folds, features, lambda s: _make_xgb(cfg, s), seed)
    out["kalman_only"] = _run_cv(panel, folds, ["e_star"], lambda s: _make_xgb(cfg, s), seed)
    out["logistic_rare_event"] = _run_cv(
        panel, folds, features, lambda s: _make_logit(cfg, s), seed, prior_tau=prior_tau
    )
    return out, folds


def fit_final_model(panel: pd.DataFrame, cfg: dict[str, Any], seed: int):
    """Fit the production model on the full training range (pre-test years)."""
    import xgboost as xgb

    folds = make_folds(panel, cfg)
    all_test_years = sorted({y for _, te in folds for y in te})
    train = panel[~panel["year"].isin(all_test_years)]
    features = list(cfg["model"]["features"])
    model = _make_xgb(cfg, seed)
    model.fit(train[features].astype(float).values, train["label"].astype(int).values)
    return model, train, features
