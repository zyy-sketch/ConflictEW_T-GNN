"""Feature engineering: Kalman signals + trend/level/growth/volatility + graph context."""
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
import pandas as pd

from .kalman import fit_kalman

# Feature columns used by the model (kept in one place for ablation studies).
FEATURE_COLS = ["e_star", "log_level", "growth", "volatility", "slope", "neighbor_e"]


def _neighbor_index(relations: pd.DataFrame) -> dict[str, list[tuple[str, float]]]:
    """Undirected adjacency from the relation table (source, target, type, weight)."""
    adj: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for _, r in relations.iterrows():
        w = float(r["weight"])
        adj[r["source"]].append((r["target"], w))
        adj[r["target"]].append((r["source"], w))
    return adj


def build_features(
    raw: dict[str, Any],
    panel: pd.DataFrame,
    cfg: dict[str, Any],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Extract Kalman signals per country and assemble model features.

    To avoid look-ahead leakage, the label at year ``t`` is predicted from
    features observed through year ``t-1``. Returns ``(panel_with_features,
    kalman_params, signals)``.
    """
    kcfg = cfg["kalman"]
    eps = float(kcfg["eps"])
    init_p = float(kcfg["init_p"])
    sigma = float(kcfg["anomaly_sigma"])

    per_country: dict[str, dict[str, Any]] = {}
    kalman_param_rows: list[dict[str, Any]] = []
    signal_rows: list[dict[str, Any]] = []

    for code, sp in raw["spending"].items():
        sp = sp.sort_values("year").reset_index(drop=True)
        yrs = sp["year"].astype(int).tolist()
        vals = sp["value"].astype(float).tolist()
        y = np.log(np.maximum(vals, eps))
        res, params = fit_kalman(
            y,
            grid_q_eta=kcfg["grid_q_eta"],
            grid_q_zeta=kcfg["grid_q_zeta"],
            grid_r_eps=kcfg["grid_r_eps"],
            init_p=init_p,
        )
        kalman_param_rows.append(
            {
                "country": code,
                "n": params["n"],
                "sigma2_eps": params["r_eps"],
                "sigma2_eta": params["q_eta"],
                "sigma2_zeta": params["q_zeta"],
                "loglik": params["ll"],
                "e_star_latest": float(res["e"][-1]),
            }
        )
        for yr, ee in zip(yrs, res["e"].tolist()):
            signal_rows.append(
                {"country": code, "year": yr, "e_star": float(ee), "anomaly": abs(float(ee)) >= sigma}
            )
        per_country[code] = {
            "year": yrs,
            "value": vals,
            "log_level": y.tolist(),
            "e": res["e"].tolist(),
            "beta": res["beta"].tolist(),
        }

    # Graph context: weighted mean of neighbours' e* (arms-race / spillover proxy).
    adj = _neighbor_index(raw["relations"])
    e_by_country_year = {
        code: dict(zip(pc["year"], pc["e"])) for code, pc in per_country.items()
    }
    neighbor_e: dict[str, dict[int, float]] = defaultdict(dict)
    for code in per_country:
        neighbors = adj.get(code, [])
        for yr in per_country[code]["year"]:
            vals = [
                (e_by_country_year[nb][yr], w)
                for nb, w in neighbors
                if yr in e_by_country_year.get(nb, {})
            ]
            if vals:
                wsum = sum(w for _, w in vals)
                neighbor_e[code][yr] = sum(e * w for e, w in vals) / wsum
            else:
                neighbor_e[code][yr] = 0.0

    rows: list[dict[str, Any]] = []
    for code, pc in per_country.items():
        year_idx = {y: i for i, y in enumerate(pc["year"])}
        for yr in pc["year"]:
            if (yr - 1) not in year_idx:  # need at least one lag
                continue
            i = year_idx[yr - 1]
            if (yr - 2) in year_idx:
                j = year_idx[yr - 2]
                growth = pc["value"][i] / pc["value"][j] - 1.0 if pc["value"][j] > 0 else 0.0
            else:
                growth = 0.0
            # 3-year rolling volatility of growth ending at t-1
            gs = []
            for k in range(max(0, i - 2), i + 1):
                if k - 1 >= 0 and pc["value"][k - 1] > 0:
                    gs.append(pc["value"][k] / pc["value"][k - 1] - 1.0)
            vol = float(np.std(gs)) if len(gs) >= 2 else 0.0
            rows.append(
                {
                    "country": code,
                    "year": yr,
                    "e_star": pc["e"][i],
                    "log_level": pc["log_level"][i],
                    "growth": growth,
                    "volatility": vol,
                    "slope": pc["beta"][i],
                    "neighbor_e": neighbor_e[code][yr - 1],
                    "anomaly_t1": float(abs(pc["e"][i]) >= sigma),
                }
            )

    feats = pd.DataFrame(rows)
    panel = panel.merge(feats, on=["country", "year"], how="inner")
    kalman_params = pd.DataFrame(kalman_param_rows)
    signals = pd.DataFrame(signal_rows)
    return panel, kalman_params, signals
