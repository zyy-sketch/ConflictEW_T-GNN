"""τ/预警预算敏感性扫描（论文 §V-B 工作点稳健性）。

对验证折 pooled 分数按"报警预算 q"（flagged 比例）取分位阈值 τ_q，
在训练期验证折与测试窗分别计算 precision/recall/Fβ=2/flagged，
并给出测试窗各预算下的 lead-time 覆盖；另附 config.yaml 发布阈值（0.7/0.5）参考行。

用法：python scripts/14_threshold_sensitivity.py
输出：artifacts/threshold_sensitivity.json / threshold_sensitivity.csv
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conflict_ew.baselines import fbeta_confusion  # noqa: E402
from conflict_ew.metrics import lead_time_analysis  # noqa: E402
from conflict_ew.paths import get_path  # noqa: E402

OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

BUDGETS = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]

oos = pd.read_csv(get_path("oos_file"))
feats = pd.read_csv(get_path("features_file"))
panel = pd.read_csv(get_path("panel_file"))
y_map = {(r["iso3"], int(r["year"])): int(r["y_onset"]) for _, r in feats.iterrows()}


def attach_y(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["y"] = [y_map.get((r["country"], int(r["year"])), np.nan) for _, r in d.iterrows()]
    return d.dropna(subset=["y"])


val = attach_y(oos[oos["fold"].str.startswith("val_fold")])
test = attach_y(oos[oos["fold"] == "test"])
assert len(val) and len(test), "oos_predictions.csv 缺少 val_fold/test 行——先运行 run_all.py"

rows = []
for q in BUDGETS:
    tau = float(np.quantile(val["p_conflict"].to_numpy(), 1 - q))
    mv = fbeta_confusion(val["y"].to_numpy().astype(int),
                         val["p_conflict"].to_numpy(), tau)
    mt = fbeta_confusion(test["y"].to_numpy().astype(int),
                         test["p_conflict"].to_numpy(), tau)
    lt = lead_time_analysis(test, panel, threshold=tau)
    rows.append({
        "mode": "budget", "budget_q": q, "tau": round(tau, 6),
        "val_precision": round(mv["precision"], 4), "val_recall": round(mv["recall"], 4),
        "val_fbeta2": round(mv["fbeta2"], 4), "val_flagged": mv["flagged"],
        "test_precision": round(mt["precision"], 4), "test_recall": round(mt["recall"], 4),
        "test_fbeta2": round(mt["fbeta2"], 4), "test_flagged": mt["flagged"],
        "test_lead_coverage": f"{lt['n_onsets_covered']}/{lt['n_onsets_total']}",
    })

# config.yaml 发布阈值参考行（非预算口径）
for name, tau in (("cfg_high", 0.7), ("cfg_mid", 0.5)):
    mv = fbeta_confusion(val["y"].to_numpy().astype(int), val["p_conflict"].to_numpy(), tau)
    mt = fbeta_confusion(test["y"].to_numpy().astype(int), test["p_conflict"].to_numpy(), tau)
    lt = lead_time_analysis(test, panel, threshold=tau)
    rows.append({
        "mode": name, "budget_q": None, "tau": tau,
        "val_precision": round(mv["precision"], 4), "val_recall": round(mv["recall"], 4),
        "val_fbeta2": round(mv["fbeta2"], 4), "val_flagged": mv["flagged"],
        "test_precision": round(mt["precision"], 4), "test_recall": round(mt["recall"], 4),
        "test_fbeta2": round(mt["fbeta2"], 4), "test_flagged": mt["flagged"],
        "test_lead_coverage": f"{lt['n_onsets_covered']}/{lt['n_onsets_total']}",
    })

res = {
    "n_val": int(len(val)), "n_test": int(len(test)),
    "grid": "按报警预算 q 在验证折 pooled 分数上取分位阈值 τ_q=Q(1-q)，同一 τ 应用到测试窗",
    "rows": rows,
}
(OUT / "threshold_sensitivity.json").write_text(
    json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
pd.DataFrame(rows).to_csv(OUT / "threshold_sensitivity.csv", index=False)
print(f"DONE: threshold_sensitivity.json/csv（{len(rows)} 行：预算网格 {len(BUDGETS)} + config 阈值 2）")
