"""Event-study alignment (paper Fig. 2c): mean e* (+/- s.e.m.) over [-5, +3] years
around the FIRST conflict onset of each of the 83 onset countries.

Output: artifacts/event_study.json / event_study_aligned.csv
        artifacts/figures/fig_event_study.png/.pdf (paper Fig. 2c)

用法：python scripts/16_event_study.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conflict_ew.build_panel import load_base_panel  # noqa: E402
from conflict_ew.paths import get_path  # noqa: E402

OUT = ROOT / "artifacts"
FIG = OUT / "figures"
FIG.mkdir(parents=True, exist_ok=True)

DELTA_MIN, DELTA_MAX = -5, 3

panel = load_base_panel()
signals = pd.read_csv(get_path("kalman_signals_file"))
feats = pd.read_csv(get_path("features_file"))

e_map = {(r["iso3"], int(r["year"])): float(r["e_star"])
         for _, r in signals.dropna(subset=["e_star"]).iterrows()}

# 首次 onset 事件：features.y_onset==1 的最早年份（每国至多一个事件）
events = (feats[feats["y_onset"] == 1]
          .groupby("iso3")["year"].min()
          .rename("t0").reset_index())

rows = []
for _, ev in events.iterrows():
    iso3, t0 = ev["iso3"], int(ev["t0"])
    for delta in range(DELTA_MIN, DELTA_MAX + 1):
        v = e_map.get((iso3, t0 + delta))
        if v is not None and np.isfinite(v):
            rows.append({"iso3": iso3, "t0": t0, "delta": delta, "e_star": v})

aligned = pd.DataFrame(rows)
agg = (aligned.groupby("delta")["e_star"]
       .agg(mean="mean", sem=lambda s: s.std(ddof=1) / np.sqrt(len(s)), n="count")
       .reset_index())
agg["mean"] = agg["mean"].round(4)
agg["sem"] = agg["sem"].round(4)

res = {
    "definition": "每个发生过至少一次 onset 的国家取首次 onset 年 t0；e* 按 delta=t−t0 对齐，"
                  "跨事件取均值与标准误（sem=std(ddof=1)/sqrt(n)）。",
    "n_events": int(len(events)),
    "n_countries_in_panel": int(panel["iso3"].nunique()),
    "window": [DELTA_MIN, DELTA_MAX],
    "lags": agg.to_dict(orient="records"),
    "note": "对应论文 Fig. 2c（event-study alignment of mean e* ± s.e.m.）。",
}
(OUT / "event_study.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                      encoding="utf-8")
aligned.to_csv(OUT / "event_study_aligned.csv", index=False)

fig, ax = plt.subplots(figsize=(4.6, 3.6))
ax.axhline(0, color="grey", lw=0.8)
ax.errorbar(agg["delta"], agg["mean"], yerr=agg["sem"], fmt="o-", color="#c0392b",
            ms=5, capsize=3, lw=1.4)
ax.axvspan(-0.5, 0.5, color="#fdebd0", zorder=0)
ax.set_xlabel("Years relative to first onset (t0)")
ax.set_ylabel("Mean standardized residual e* (± s.e.m.)")
ax.set_title(f"Event-study alignment of Kalman residuals (n = {len(events)} first onsets)")
ax.set_xticks(range(DELTA_MIN, DELTA_MAX + 1))
fig.savefig(FIG / "fig_event_study.png", bbox_inches="tight")
fig.savefig(FIG / "fig_event_study.pdf", bbox_inches="tight")
plt.close(fig)
print(f"DONE: event_study.json/csv + fig_event_study（{len(events)} 个首次 onset 事件）")
