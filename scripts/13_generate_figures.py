"""论文级图表生成（一命令复现论文全部插图的数据版本）。

输入：artifacts/ 下既有产物（oos_predictions / metrics / calibration_reliability /
      threshold_sensitivity）+ data/features.csv
输出：artifacts/figures/fig{1..5}_*.png(300dpi)/.pdf

注意：图内文字为英文（避免 matplotlib 默认字体缺中文字形）；
     依赖 run_all.py 与 14_threshold_sensitivity.py 先行产出数据。
用法：python scripts/13_generate_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
FIG = ART / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})

GOLDEN_AUC = 0.5447


def _save(fig, name: str) -> None:
    fig.savefig(FIG / f"{name}.png", bbox_inches="tight")
    fig.savefig(FIG / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {name}.png/.pdf")


metrics = json.loads((ART / "metrics.json").read_text(encoding="utf-8"))
oos = pd.read_csv(ART / "oos_predictions.csv")
feats = pd.read_csv(ROOT / "data" / "features.csv")
y_map = {(r["iso3"], int(r["year"])): int(r["y_onset"]) for _, r in feats.iterrows()}

test = oos[oos["fold"] == "test"].copy()
test["y"] = [y_map.get((r["country"], int(r["year"])), np.nan) for _, r in test.iterrows()]
test = test.dropna(subset=["y"])
y_te = test["y"].to_numpy().astype(int)
p_te = test["p_conflict"].to_numpy()

# ---- fig1: ROC ----
fpr, tpr, _ = roc_curve(y_te, p_te)
auc = metrics["auc"]["auc"]
lo, hi = metrics["auc"]["ci_lower"], metrics["auc"]["ci_upper"]
fig, ax = plt.subplots(figsize=(4.2, 3.8))
ax.plot(fpr, tpr, color="#c0392b", lw=1.6,
        label=f"KF+MF (AUC = {auc:.3f}, DeLong 95% CI [{lo:.3f}, {hi:.3f}])")
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="chance")
ax.set_xlabel("False positive rate")
ax.set_ylabel("True positive rate")
ax.set_title("ROC — held-out 2015–2024 (n = 1,650, 61 onsets)")
ax.legend(loc="lower right", fontsize=8)
_save(fig, "fig1_roc")

# ---- fig2: 可靠性图（KF+MF raw 8-bin + Wilson 95% CI + isotonic 校准后）----
from scipy.stats import norm  # noqa: E402

_z = norm.ppf(0.975)


def _wilson_ci(n: int, p_hat: float) -> tuple[float, float]:
    """二项比例的 Wilson 95% 置信区间（小样本 bin 的诚实不确定度）。"""
    if n == 0:
        return float("nan"), float("nan")
    denom = 1 + _z**2 / n
    center = (p_hat + _z**2 / (2 * n)) / denom
    half = _z * np.sqrt(p_hat * (1 - p_hat) / n + _z**2 / (4 * n**2)) / denom
    return max(center - half, 0.0), min(center + half, 1.0)


cal = json.loads((ART / "calibration_reliability.json").read_text(encoding="utf-8"))["models"]["KF+MF"]
raw_bins = [b for b in cal["raw"]["bins"] if b["n"] > 0]
fig, ax = plt.subplots(figsize=(4.2, 3.8))
ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect calibration")
for b in raw_bins:
    ci_lo, ci_hi = _wilson_ci(b["n"], b["obs"])
    ax.errorbar(b["pred"], b["obs"], yerr=[[b["obs"] - ci_lo], [ci_hi - b["obs"]]],
                fmt="o", color="#c0392b", ms=6, capsize=3,
                label="raw score (8 equal-width bins, Wilson 95% CI)" if b is raw_bins[0] else None)
    ax.annotate(f"n={b['n']}", (b["pred"], b["obs"]), textcoords="offset points",
                xytext=(8, 4), fontsize=7, color="#555555")
cal_bins = [b for b in cal["isotonic_calibrated"]["bins"] if b["n"] > 0]
for j, b in enumerate(cal_bins):
    ci_lo, ci_hi = _wilson_ci(b["n"], b["obs"])
    ax.errorbar(b["pred"], b["obs"], yerr=[[b["obs"] - ci_lo], [ci_hi - b["obs"]]],
                fmt="s", color="#2c7fb8", ms=6, capsize=3,
                label="after isotonic recalibration" if j == 0 else None)
ax.set_xlim(-0.03, 1.0)
ax.set_ylim(-0.03, 1.0)
ax.set_xlabel("Mean predicted conflict probability")
ax.set_ylabel("Observed onset frequency")
ax.set_title("Reliability — raw Brier 0.0391 → calibrated 0.0370")
ax.legend(loc="upper left", fontsize=8)
_save(fig, "fig2_reliability")

# ---- fig3: 消融 ----
abl = metrics["ablation"]
names = [a["ablation"] for a in abl]
vals = [a["test_auc"] for a in abl]
fig, ax = plt.subplots(figsize=(5.6, 3.8))
bars = ax.bar(range(len(vals)), vals, color="#7fb3d5", edgecolor="#34495e")
ax.axhline(GOLDEN_AUC, color="#c0392b", ls="--", lw=1.2,
           label=f"full model ({GOLDEN_AUC:.4f})")
for i, (b, v) in enumerate(zip(bars, vals)):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.008, f"{v:.4f}", ha="center", fontsize=8)
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=20, ha="right", fontsize=8)
ax.set_ylim(0.4, 0.75)
ax.set_ylabel("Held-out AUC (2015–2024)")
ax.set_title("Ablation: dropping the military-anomaly signal F1 raises AUC")
ax.legend(fontsize=8)
_save(fig, "fig3_ablation")

# ---- fig4: 基线配对差森林图 ----
comp = [d for d in metrics["baseline_comparison"] if d.get("diff") is not None]
fig, ax = plt.subplots(figsize=(5.6, 3.2))
for i, d in enumerate(comp):
    sig = "★" if d.get("fdr_significant") else "n.s."
    color = "#c0392b" if d.get("fdr_significant") else "#7f8c8d"
    ax.errorbar(d["diff"], i, xerr=[[d["diff"] - d["ci_lower"]], [d["ci_upper"] - d["diff"]]],
                fmt="o", color=color, capsize=4, ms=6)
    ax.text(d["ci_upper"] + 0.006, i, f"p={d['p_value']:.3f} (FDR {sig})", va="center", fontsize=8)
ax.axvline(0, color="grey", ls="--", lw=1)
ax.set_yticks(range(len(comp)))
ax.set_yticklabels([f"KF+MF − {d['baseline']}" for d in comp], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel("Paired AUC difference (KF+MF − baseline), 1,000-resample bootstrap 95% CI")
ax.set_title("Benchmark comparison — held-out 2015–2024")
_save(fig, "fig4_baseline_forest")

# ---- fig5: τ/预算敏感性（依赖 14_threshold_sensitivity.py 产物）----
ts_path = ART / "threshold_sensitivity.json"
if ts_path.exists():
    rows = [r for r in json.loads(ts_path.read_text(encoding="utf-8"))["rows"]
            if r["mode"] == "budget"]
    qs = [r["budget_q"] for r in rows]
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6), sharex=True)
    for ax, pre, title in ((axes[0], "val", "Training validation folds"),
                           (axes[1], "test", "Held-out test window (same τ applied)")):
        ax.plot(qs, [r[f"{pre}_precision"] for r in rows], "o-", color="#c0392b", ms=4,
                label="precision")
        ax.plot(qs, [r[f"{pre}_recall"] for r in rows], "s-", color="#2c7fb8", ms=4,
                label="recall")
        ax.set_xscale("log")
        ax.set_xlabel("Alert budget q (flagged fraction)")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("Precision / recall at τ_q")
    axes[1].legend(fontsize=8)
    fig.suptitle("Threshold/budget sensitivity — Fβ=2-optimal rule is budget-dependent", y=1.02)
    _save(fig, "fig5_threshold_sensitivity")
else:
    print("  - threshold_sensitivity.json 不存在，跳过 fig5（先运行 scripts/14_threshold_sensitivity.py）")

print(f"DONE: 图表已写入 {FIG}")
