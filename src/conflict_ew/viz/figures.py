"""Publication-quality vector figures (matplotlib).

Color-blind-safe (ColorBrewer RdYlBu), Arial font, 300 DPI, PDF/SVG output to
match journal requirements (SCI_OPTIMIZATION_PLAN.md §C). All figure text is
English so no CJK font embedding is required.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

# ColorBrewer RdYlBu (color-blind safe)
RDYLBU = ["#2166ac", "#67a9cf", "#d1e5f0", "#fddbc7", "#ef8a62", "#b2182b"]
ACCENT = "#b2182b"
GRID = "#cccccc"

plt.rcParams.update(
    {
        "font.family": "Arial",
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "axes.edgecolor": "#333333",
        "axes.labelcolor": "#111111",
        "xtick.color": "#111111",
        "ytick.color": "#111111",
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.6,
        "figure.dpi": 110,
    }
)


def _size(width_mm: float, ratio: float) -> tuple[float, float]:
    w = width_mm / 25.4
    return (w, w * ratio)


def _save(fig: Any, path: Path, cfg: dict[str, Any]) -> str:
    fmt = cfg["figures"]["format"]
    dpi = int(cfg["figures"]["dpi"])
    out = path.with_suffix(f".{fmt}")
    fig.savefig(out, dpi=dpi, bbox_inches="tight")
    # always also emit a PNG preview at 300 dpi
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_roc(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    auc: float,
    ci: tuple[float, float],
    path: Path,
    cfg: dict[str, Any],
) -> str:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig, ax = plt.subplots(figsize=_size(float(cfg["figures"]["width_mm"]), 0.8))
    ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1, label="Chance")
    ax.plot(fpr, tpr, color=ACCENT, lw=2, label=f"AUC = {auc:.3f} (95% CI {ci[0]:.3f}–{ci[1]:.3f})")
    ax.fill_between(fpr, tpr, alpha=0.10, color=ACCENT)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (out-of-sample)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return _save(fig, path, cfg)


def plot_reliability(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
    path: Path,
    cfg: dict[str, Any],
) -> str:
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    centers, obs, preds = [], [], []
    for k in range(n_bins):
        if k == n_bins - 1:
            mask = (y_prob >= bins[k]) & (y_prob <= bins[k + 1])
        else:
            mask = (y_prob >= bins[k]) & (y_prob < bins[k + 1])
        if mask.sum() == 0:
            continue
        centers.append((bins[k] + bins[k + 1]) / 2)
        preds.append(float(y_prob[mask].mean()))
        obs.append(float(y_true[mask].mean()))
    fig, ax = plt.subplots(figsize=_size(float(cfg["figures"]["width_mm"]), 0.8))
    ax.plot([0, 1], [0, 1], "--", color="#888888", lw=1, label="Perfect calibration")
    ax.plot(centers, obs, "o-", color=ACCENT, lw=1.5, ms=4, label="Model")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted probability")
    ax.set_ylabel("Observed frequency")
    ax.set_title("Reliability Diagram (Brier decomposition)")
    ax.legend(loc="lower right", frameon=False)
    fig.tight_layout()
    return _save(fig, path, cfg)


def plot_kalman_signal(
    years: list[int],
    e_star: np.ndarray,
    sigma: float,
    country: str,
    path: Path,
    cfg: dict[str, Any],
) -> str:
    years = np.asarray(years)
    fig, ax = plt.subplots(figsize=_size(float(cfg["figures"]["width_mm"]), 0.55))
    ax.plot(years, e_star, color=RDYLBU[0], lw=1.2, label="e* (standardized residual)")
    ax.axhline(sigma, color=ACCENT, ls="--", lw=1, label=f"±{sigma}σ threshold")
    ax.axhline(-sigma, color=ACCENT, ls="--", lw=1)
    ax.axhline(0, color="#555555", lw=0.6)
    ax.fill_between(years, -sigma, sigma, color="#000000", alpha=0.03)
    ax.set_xlabel("Year")
    ax.set_ylabel("Standardized residual e*")
    ax.set_title(f"Kalman-filter military-expenditure anomaly signal — {country}")
    ax.legend(loc="upper left", frameon=False)
    fig.tight_layout()
    return _save(fig, path, cfg)


def plot_feature_importance(
    names: list[str],
    importances: np.ndarray,
    path: Path,
    cfg: dict[str, Any],
) -> str:
    order = np.argsort(importances)
    fig, ax = plt.subplots(figsize=_size(float(cfg["figures"]["width_mm"]), 0.6))
    ax.barh([names[i] for i in order], importances[order], color=RDYLBU[1])
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature importance (SHAP)")
    fig.tight_layout()
    return _save(fig, path, cfg)


def plot_ablation(
    labels: list[str],
    aucs: list[float],
    stds: list[float],
    path: Path,
    cfg: dict[str, Any],
) -> str:
    fig, ax = plt.subplots(figsize=_size(float(cfg["figures"]["width_mm"]), 0.55))
    y = np.arange(len(labels))[::-1]
    colors = [ACCENT if l == labels[0] else RDYLBU[1] for l in labels]
    ax.barh(y, aucs, xerr=stds, color=colors, capsize=3, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Out-of-sample AUC (mean ± SD across folds)")
    ax.set_xlim(0, 1)
    ax.set_title("Model ablation")
    fig.tight_layout()
    return _save(fig, path, cfg)
