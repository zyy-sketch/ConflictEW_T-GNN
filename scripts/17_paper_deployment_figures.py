"""Paper deployment figures (Fig.2a/b, Fig.5) + country code map + SHAP figures.

C1 rule: panel start == 1992 & max mean log milex (fallback [1990,1992]).
Fig.5: top-5 by mean raw test score, T1-T5; lines 0.7/0.5/1-3.
Map: artifacts/country_code_map.csv.
用法：python scripts/17_paper_deployment_figures.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conflict_ew.build_panel import load_base_panel  # noqa: E402
from conflict_ew.paths import get_path  # noqa: E402

OUT = ROOT / "artifacts"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 9})

panel = load_base_panel()
signals = pd.read_csv(get_path("kalman_signals_file"))
oos = pd.read_csv(get_path("oos_file"))
shap_vals = pd.read_csv(OUT / "shap_values.csv")
shap_imp = pd.read_csv(OUT / "shap_importance.csv")

e_map = {(r["iso3"], int(r["year"])): float(r["e_star"])
         for _, r in signals.dropna(subset=["e_star"]).iterrows()}

map_rows = []
top5 = []

# ---- C1 选取（确定性规则，规则写入映射文件）----
start_year = panel.groupby("iso3")["year"].min().rename("start")
mean_log = panel.groupby("iso3")["mil_expense_log"].mean().rename("mean_log_milex")
cand = pd.concat([start_year, mean_log], axis=1)
pool = cand[cand["start"] == 1992]
fallback = False
if pool.empty:
    pool = cand[(cand["start"] >= 1990) & (cand["start"] <= 1992)]
    fallback = True
c1_iso3 = pool.sort_values("mean_log_milex", ascending=False).index[0]
c1_start = int(cand.loc[c1_iso3, "start"])
map_rows.append({"code": "C1", "iso3": c1_iso3,
                 "role": "Fig.2 representative major spender",
                 "detail": f"panel start {c1_start} (fallback={fallback})"})

# ---- Fig. 2a：C1 军费对数序列 + |e*|≥2 橙圈 ----
g = panel[panel["iso3"] == c1_iso3].sort_values("year")
sig1 = signals[signals["iso3"] == c1_iso3].set_index("year")
anom_years = [int(y) for y in sig1.index if abs(sig1.loc[y, "e_star"]) >= 2]
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(g["year"], g["mil_expense_log"], color="#2c3e50", lw=1.5)
anom = g[g["year"].isin(anom_years)]
ax.scatter(anom["year"], anom["mil_expense_log"], facecolors="none",
           edgecolors="#e67e22", s=52, lw=1.6, label=r"|e*| $\geq$ 2")
ax.set_xlabel("Year")
ax.set_ylabel("log military expenditure (current US$)")
ax.set_title(f"Representative major spender (C1), data from {c1_start}")
ax.legend(fontsize=8)
fig.savefig(FIG / "fig_paper2a_milex.png", bbox_inches="tight")
fig.savefig(FIG / "fig_paper2a_milex.pdf", bbox_inches="tight")
plt.close(fig)
print("  ✓ fig_paper2a_milex")

# ---- Fig. 2b：C1 标准化残差 + ±2 带 ----
e1 = sig1.reset_index()
fig, ax = plt.subplots(figsize=(5.2, 3.6))
ax.plot(e1["year"], e1["e_star"], color="#2c3e50", lw=1.2)
ax.axhline(2, color="#c0392b", ls="--", lw=1)
ax.axhline(-2, color="#c0392b", ls="--", lw=1)
ax.set_xlabel("Year")
ax.set_ylabel("Standardized one-step-ahead residual e*")
ax.set_title("C1 standardized residual e* (±2 significance band)")
fig.savefig(FIG / "fig_paper2b_residual.png", bbox_inches="tight")
fig.savefig(FIG / "fig_paper2b_residual.pdf", bbox_inches="tight")
plt.close(fig)
print("  ✓ fig_paper2b_residual")

# ---- Fig. 5：测试窗 top-5 轨迹（T1–T5）----
test = oos[oos["fold"] == "test"].copy()
mean_p = (test.groupby("country")["p_conflict"].mean()
          .sort_values(ascending=False).head(5))
top5 = list(mean_p.index)
fig, ax = plt.subplots(figsize=(5.6, 3.8))
markers = ["o", "s", "^", "D", "v"]
for rank, iso3 in enumerate(top5, start=1):
    code = f"T{rank}"
    d = test[test["country"] == iso3].sort_values("year")
    ax.plot(d["year"], d["p_conflict"], marker=markers[rank - 1], ms=4, lw=1.3,
            label=code)
    map_rows.append({"code": code, "iso3": iso3, "role": "Fig.5 trajectory rank",
                     "detail": f"rank {rank}, mean raw {mean_p[iso3]:.4f}"})
ax.axhline(0.7, color="#c0392b", ls="--", lw=1.1, label="tier-1 (0.7)")
ax.axhline(0.5, color="#e67e22", ls=":", lw=1.1, label="mid (0.5)")
ax.axhline(1 / 3, color="#7f8c8d", ls="-.", lw=1.0, label="isotonic constant (1/3)")
ax.set_ylim(0, 1.02)
ax.set_xlabel("Year")
ax.set_ylabel("Raw (pre-calibration) score p")
ax.set_title("Five highest-scoring countries, held-out 2015–2024")
ax.legend(fontsize=7, ncol=2)
fig.savefig(FIG / "fig5_trajectories.png", bbox_inches="tight")
fig.savefig(FIG / "fig5_trajectories.pdf", bbox_inches="tight")
plt.close(fig)
print("  ✓ fig5_trajectories")

# ---- SHAP：全局重要性 + 因子分布（论文 Fig. 3 数据版）----
imp = shap_imp.sort_values("mean_abs")
fig, ax = plt.subplots(figsize=(4.8, 3.2))
ax.barh(imp["factor_id"], imp["mean_abs"], color="#7fb3d5", edgecolor="#34495e")
ax.set_xlabel("mean |SHAP value|")
ax.set_title("Global factor importance")
fig.savefig(FIG / "fig_shap_importance.png", bbox_inches="tight")
fig.savefig(FIG / "fig_shap_importance.pdf", bbox_inches="tight")
plt.close(fig)
print("  ✓ fig_shap_importance")

order = list(imp.sort_values("mean_abs", ascending=False)["factor_id"])
fig, ax = plt.subplots(figsize=(5.6, 3.4))
data = [shap_vals.loc[shap_vals["factor_id"] == f, "shap_value"].to_numpy() for f in order]
ax.boxplot(data, tick_labels=order, showfliers=False)
ax.axhline(0, color="grey", lw=0.8)
ax.set_ylabel("SHAP value")
ax.set_title("Per-factor SHAP distribution (test window)")
fig.savefig(FIG / "fig_shap_distribution.png", bbox_inches="tight")
fig.savefig(FIG / "fig_shap_distribution.pdf", bbox_inches="tight")
plt.close(fig)
print("  ✓ fig_shap_distribution")

# ---- 国家脱敏编码映射（论文 Fig.5 图注所指文件）----
pd.DataFrame(map_rows).to_csv(OUT / "country_code_map.csv", index=False)
print("  ✓ country_code_map.csv")
