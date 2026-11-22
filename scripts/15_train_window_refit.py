"""训练窗重估端到端主结果（审稿攻击面加固：Kalman 全序列 MLE 例外的对照版）。

将每国 Kalman 超参改为仅在训练窗(<=2014) MLE 估计、冻结后前向递归生成 e*，
随后走完整下游（多因子特征 → blocked CV XGBoost → 测试窗评估），
与发布版（全序列估参，AUC=0.5447）对比——直接量化该已披露例外对最终结论的影响。

用法：python scripts/15_train_window_refit.py
输出：artifacts/m2_refit_end_to_end.json
注意：train_and_predict 会覆写 artifacts/xgb_model.json 等文件，本脚本
     对这些路径做字节级快照并在结束后原样恢复（发布工件不受影响）。
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conflict_ew.baselines import delong_ci  # noqa: E402
from conflict_ew.build_panel import load_base_panel  # noqa: E402
from conflict_ew.external import ingest_external  # noqa: E402
from conflict_ew.factors import build_features  # noqa: E402
from conflict_ew.kalman import _build_grid, _fit_single, _kalman_filter  # noqa: E402
from conflict_ew.paths import get_path  # noqa: E402
from conflict_ew.xgb import train_and_predict  # noqa: E402

OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

panel = load_base_panel()
sig_full = pd.read_csv(get_path("kalman_signals_file"))

# ---- 1) 训练窗(<=2014) MLE 估参 + 冻结前推（与 09_window_sensitivity.py 同式）----
grid = _build_grid()
rows = []
for iso3, grp in panel.groupby("iso3"):
    g = grp.sort_values("year").copy()
    y_all = g["mil_expense_log"].to_numpy(dtype=float)
    yrs = g["year"].to_numpy()
    train_mask = yrs <= 2014
    y_tr = y_all[train_mask]
    if len(y_tr) < 8:
        continue
    fit = _fit_single(iso3, y_tr, grid)
    if not np.isfinite(fit.log_lik):
        continue
    valid = ~np.isnan(y_all)
    e, F, _, _, _ = _kalman_filter(y_all[valid], fit.sigma2_eps, fit.sigma2_eta, fit.sigma2_zeta)
    std_e = np.clip(e / np.sqrt(np.maximum(F, 1e-12)), -10.0, 10.0)
    full = np.full(len(y_all), np.nan)
    full[valid] = std_e
    for yr, ev in zip(yrs, full):
        rows.append({"iso3": iso3, "year": int(yr), "e_star_refit": float(ev)})

sig2 = sig_full.merge(pd.DataFrame(rows), on=["iso3", "year"], how="inner")
sig2["e_star"] = sig2["e_star_refit"]
sig2["is_anomaly"] = (sig2["e_star"].abs() >= 2).astype(int)
sig2 = sig2[["iso3", "year", "e_star", "is_anomaly"]]

# ---- 2) 受保护工件快照（train_and_predict 会覆写模型/OOS 文件）----
protected = ["model_file", "params_file", "oos_file", "shap_file"]
snapshot = {}
for key in protected:
    p = get_path(key)
    if p.exists():
        snapshot[p] = p.read_bytes()

try:
    # ---- 3) 完整下游：特征 → blocked CV XGBoost → 测试窗 ----
    ext = ingest_external()
    features = build_features(panel, sig2, ext.adjacency, out_path=None)
    res = train_and_predict(features)
    oos = res["oos_df"]
    test = oos[oos["fold"] == "test"].copy()
    y_map = {(r["iso3"], int(r["year"])): int(r["y_onset"]) for _, r in features.iterrows()}
    test["y"] = [y_map.get((r["country"], int(r["year"])), np.nan) for _, r in test.iterrows()]
    test = test.dropna(subset=["y"])
    y_te = test["y"].to_numpy().astype(int)
    p_te = test["p_conflict"].to_numpy()
    _, lo, hi = delong_ci(y_te, p_te)
    from sklearn.metrics import roc_auc_score  # noqa: E402

    refit_auc = float(roc_auc_score(y_te, p_te))
    refit_brier = float(np.mean((y_te - p_te) ** 2))
finally:
    # ---- 4) 恢复受保护工件（字节级还原）----
    for p, data in snapshot.items():
        p.write_bytes(data)

released = json.loads((OUT / "metrics.json").read_text(encoding="utf-8"))
rel_auc = released["auc"]["auc"]

out = {
    "finding": "Kalman 超参改为仅训练窗(<=2014) MLE 估计、冻结后前推，端到端重跑"
               "特征/模型/评估；与发布版（全序列 MLE）对比该已披露例外的影响。",
    "n_countries_refit": int(pd.DataFrame(rows)["iso3"].nunique()),
    "n_test": int(len(y_te)),
    "n_test_positive": int(y_te.sum()),
    "refit_auc": round(refit_auc, 4),
    "refit_auc_ci_delong": [round(lo, 4), round(hi, 4)],
    "refit_brier_raw": round(refit_brier, 4),
    "released_auc": round(rel_auc, 4),
    "delta_auc_vs_released": round(refit_auc - rel_auc, 4),
    "interpretation": "若 |delta| 小于 DeLong 半宽，说明该例外的披露不影响主结论方向；"
                      "本文以敏感性分析（M2：r=0.926/flag 一致率 98.9%）作为稳健性证据。",
}
(OUT / "m2_refit_end_to_end.json").write_text(
    json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(out, ensure_ascii=False, indent=1))
print("DONE: m2_refit_end_to_end.json（受保护工件已恢复）")
