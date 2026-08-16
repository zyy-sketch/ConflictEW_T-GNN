"""End-to-end pipeline orchestrator (config-driven, seed-fixed, provenance-logged).

Stage order mirrors SCI_OPTIMIZATION_PLAN.md §A1:

    load -> features -> train -> eval -> figures

Each stage reads ``config.yaml``, fixes the RNG, and writes both the data
artifacts and a JSONL run-log (git hash + software versions + seed + timestamp).
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .config import load_config, resolve
from .data import build_panel, load_raw, write_manifest
from .eval import metrics
from .features.factors import build_features
from .models import xgb as xgb_mod
from .viz import figures


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import xgboost as xgb

        xgb.set_config(verbosity=0)
    except Exception:
        pass


def _art(cfg: dict[str, Any]) -> Path:
    p = Path(cfg["paths"]["artifacts"])
    p.mkdir(parents=True, exist_ok=True)
    return p


def stage_load(cfg: dict[str, Any]) -> dict[str, Any]:
    from .meta import write_runlog

    raw = load_raw(cfg)
    panel = build_panel(raw)
    manifest = write_manifest(raw, cfg, panel)
    _art(cfg).joinpath("panel.csv").write_text(panel.to_csv(index=False), encoding="utf-8")
    write_runlog(cfg, "load", {"manifest": manifest})
    return {"raw": raw, "panel": panel, "manifest": manifest}


def stage_features(cfg: dict[str, Any]) -> dict[str, Any]:
    from .meta import write_runlog

    raw = load_raw(cfg)
    panel = build_panel(raw)
    panel, kalman_params, signals = build_features(raw, panel, cfg)
    art = _art(cfg)
    panel.to_csv(art / "features.csv", index=False)
    kalman_params.to_csv(art / "kalman_params.csv", index=False)
    signals.to_csv(art / "kalman_signals.csv", index=False)
    write_runlog(
        cfg,
        "features",
        {
            "feature_columns": list(cfg["model"]["features"]),
            "n_countries": int(panel["country"].nunique()),
            "n_rows": int(len(panel)),
            "positive_labels": int(panel["label"].sum()),
        },
    )
    return {"panel": panel, "kalman_params": kalman_params, "signals": signals}


def stage_train(cfg: dict[str, Any]) -> dict[str, Any]:
    from .meta import write_runlog

    panel = pd.read_csv(_art(cfg) / "features.csv")
    seed = int(cfg["reproducibility"]["seed"])
    oos, folds = xgb_mod.run_all_models(panel, cfg, seed)

    # persist OOS predictions (one file per model, plus a combined file)
    combined = []
    for name, df in oos.items():
        df = df.copy()
        df["model"] = name
        combined.append(df)
        (_art(cfg) / f"oos_{name}.csv").write_text(df.to_csv(index=False), encoding="utf-8")
    pd.concat(combined).to_csv(_art(cfg) / "oos_predictions.csv", index=False)

    fold_desc = [{"fold": i, "train_start": tr[0], "train_end": tr[-1], "test": te} for i, (tr, te) in enumerate(folds)]
    (_art(cfg) / "xgb_params.json").write_text(
        json.dumps(
            {
                "model": cfg["model"]["name"],
                "params": cfg["model"]["params"],
                "rare_event": cfg["model"]["rare_event"],
                "features": cfg["model"]["features"],
                "folds": fold_desc,
                "seed": seed,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_runlog(cfg, "train", {"folds": fold_desc, "models": list(oos.keys())})
    return {"oos": oos, "folds": folds}


def _pooled_metrics(oos: pd.DataFrame) -> dict[str, Any]:
    y = oos["y_true"].to_numpy()
    p = oos["y_prob"].to_numpy()
    auc, auc_ci = metrics.auc_delong_ci(y, p)
    b, b_ci = metrics.brier_bootstrap_ci(y, p, n_boot=1000, seed=0)
    rel, res, unc = metrics.brier_decomposition(y, p, n_bins=10)
    return {
        "n": int(len(y)),
        "positives": int(y.sum()),
        "base_rate": float(y.mean()),
        "auc": auc,
        "auc_ci_95": list(auc_ci),
        "brier": b,
        "brier_ci_95": list(b_ci),
        "brier_reliability": rel,
        "brier_resolution": res,
        "brier_uncertainty": unc,
    }


def stage_eval(cfg: dict[str, Any]) -> dict[str, Any]:
    from .meta import write_runlog

    oos = pd.read_csv(_art(cfg) / "oos_predictions.csv")
    seed = int(cfg["reproducibility"]["seed"])
    alpha_fdr = float(cfg["eval"]["alpha_fdr"])

    results: dict[str, Any] = {}
    for name, sub in oos.groupby("model"):
        results[name] = _pooled_metrics(sub)

    # fold-level AUC (mean ± SD) for ablation plot
    ablation = {}
    for name, sub in oos.groupby("model"):
        fold_aucs = []
        for _, fsub in sub.groupby("fold"):
            if fsub["y_true"].nunique() < 2:
                continue
            a = metrics.auc(fsub["y_true"], fsub["y_prob"])
            if np.isfinite(a):
                fold_aucs.append(a)
        ablation[name] = {
            "auc": float(np.mean(fold_aucs)) if fold_aucs else float("nan"),
            "auc_sd": float(np.std(fold_aucs)) if fold_aucs else float("nan"),
            "n_folds": len(fold_aucs),
        }

    # paired comparison: full XGB vs each baseline (DeLong test + paired bootstrap)
    base = oos[oos["model"] == "xgb"]
    comparisons: dict[str, Any] = {}
    raw_pvalues: dict[str, float] = {}
    for name in results:
        if name == "xgb":
            continue
        sub = oos[oos["model"] == name]
        m = base.merge(sub, on=["country", "year"], suffixes=("_xgb", "_b"))
        if len(m) and m["y_true_xgb"].nunique() >= 2:
            y = m["y_true_xgb"].to_numpy()
            pa = m["y_prob_xgb"].to_numpy()
            pb = m["y_prob_b"].to_numpy()
            delong = metrics.delong_test(y, pa, pb)
            diff, pval = metrics.paired_bootstrap_pvalue(y, pa, pb, n_boot=1000, seed=seed)
            comparisons[name] = {
                "auc_diff_bootstrap": diff,
                "p_bootstrap": pval,
                "delong": delong,
            }
            raw_pvalues[name] = pval

    # Benjamini-Hochberg FDR correction across baseline comparisons (§B2)
    if raw_pvalues:
        names_order = list(raw_pvalues.keys())
        adj, rej = metrics.benjamini_hochberg(
            [raw_pvalues[n] for n in names_order], alpha=alpha_fdr
        )
        for i, n in enumerate(names_order):
            comparisons[n]["p_bootstrap_fdr"] = float(adj[i])
            comparisons[n]["rejected_fdr"] = bool(rej[i])

    out = {
        "confidence": cfg["eval"]["confidence"],
        "models": results,
        "ablation": ablation,
        "paired_vs_xgb": comparisons,
    }
    (_art(cfg) / "metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    write_runlog(cfg, "eval", out)
    return out


def stage_figures(cfg: dict[str, Any]) -> dict[str, Any]:
    from .meta import write_runlog

    art = _art(cfg)
    fig_dir = Path(cfg["paths"]["figures"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    oos = pd.read_csv(art / "oos_predictions.csv")
    metrics_json = json.loads((art / "metrics.json").read_text(encoding="utf-8"))
    signals = pd.read_csv(art / "kalman_signals.csv")

    paths: dict[str, str] = {}

    sub = oos[oos["model"] == "xgb"]
    m = metrics_json["models"]["xgb"]
    paths["roc"] = figures.plot_roc(
        sub["y_true"].to_numpy(), sub["y_prob"].to_numpy(), m["auc"], tuple(m["auc_ci_95"]),
        fig_dir / "fig_roc", cfg,
    )
    paths["reliability"] = figures.plot_reliability(
        sub["y_true"].to_numpy(), sub["y_prob"].to_numpy(), int(cfg["eval"]["brier_bins"]),
        fig_dir / "fig_reliability", cfg,
    )

    # Kalman signal for a few illustrative countries (top |e*| latest)
    top = (
        signals.groupby("country")["e_star"].last().abs().sort_values(ascending=False).head(3).index.tolist()
    )
    for code in top:
        s = signals[signals["country"] == code].sort_values("year")
        paths[f"kalman_{code}"] = figures.plot_kalman_signal(
            s["year"].tolist(), s["e_star"].to_numpy(), float(cfg["kalman"]["anomaly_sigma"]),
            code, fig_dir / f"fig_kalman_{code}", cfg,
        )

    # ablation bar
    abl = metrics_json["ablation"]
    names = list(abl.keys())
    aucs = [abl[n]["auc"] for n in names]
    stds = [abl[n]["auc_sd"] for n in names]
    paths["ablation"] = figures.plot_ablation(names, aucs, stds, fig_dir / "fig_ablation", cfg)

    # SHAP feature importance (from a final model, if available)
    try:
        panel = pd.read_csv(art / "features.csv")
        model, train, feats = xgb_mod.fit_final_model(panel, cfg, int(cfg["reproducibility"]["seed"]))
        import shap

        explainer = shap.TreeExplainer(model)
        X = train[feats].astype(float)
        shv = explainer.shap_values(X)
        if isinstance(shv, list):
            shv = shv[1] if len(shv) > 1 else shv[0]
        paths["shap"] = figures.plot_feature_importance(feats, np.abs(shv).mean(axis=0), fig_dir / "fig_shap", cfg)
    except Exception as exc:  # SHAP / xgboost optional
        paths["shap"] = f"skipped: {exc}"

    write_runlog(cfg, "figures", {"paths": paths})
    return paths


def run_all(cfg_path: str | Path = "config.yaml") -> None:
    cfg = load_config(cfg_path)
    seed_everything(int(cfg["reproducibility"]["seed"]))
    stage_load(cfg)
    stage_features(cfg)
    stage_train(cfg)
    stage_eval(cfg)
    stage_figures(cfg)


if __name__ == "__main__":
    import sys

    run_all(sys.argv[1] if len(sys.argv) > 1 else "config.yaml")
    print("pipeline complete")
