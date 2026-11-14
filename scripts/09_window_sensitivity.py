# -*- coding: utf-8 -*-
"""Kalman 估计窗口敏感性（M2）+ e* 价格诊断（M3）——论文 §III-A / §IV 局限性口径。

规范来源：论文 revision_v3 release_package scripts/m2_m3_calibration_checks.py
（golden：pooled r=0.926、flag 一致率 0.9886、inflation 相关 0.046/0.045）。

M2: 用训练窗(<=2014)估参、冻结超参后前向递归到 2024，与管线原样（全序列
    估参）的 e* 比较——量化"信号层估计窗含测试期信息"的敏感性（论文 §V-A 披露）。
M3: e* 与 WDI 通胀率的相关性（pooled + 组内），验证 e* 非价格噪声驱动。

输出：artifacts/m2_window_sensitivity.json、m3_price_diagnostic.json、
      m2_per_country_comparison.csv
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conflict_ew.kalman import _build_grid, _fit_single, _kalman_filter  # noqa: E402
from conflict_ew.paths import get_path  # noqa: E402

OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

panel = pd.read_csv(get_path("panel_file"))
sig_full = pd.read_csv(get_path("kalman_signals_file"))
params_full = pd.read_csv(get_path("kalman_params_file"))

# ============ M2: 训练窗估参 + 冻结前推 ============
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
    # 冻结超参，在完整序列上做一步预测残差（同一递归规则）
    valid = ~np.isnan(y_all)
    e, F, _, _, _ = _kalman_filter(y_all[valid], fit.sigma2_eps, fit.sigma2_eta, fit.sigma2_zeta)
    std_e = np.clip(e / np.sqrt(np.maximum(F, 1e-12)), -10.0, 10.0)
    full = np.full(len(y_all), np.nan)
    full[valid] = std_e
    for yr, ev in zip(yrs, full):
        rows.append({"iso3": iso3, "year": int(yr), "e_star_trainwindow": float(ev),
                     "sigma2_eps": fit.sigma2_eps, "sigma2_eta": fit.sigma2_eta,
                     "sigma2_zeta": fit.sigma2_zeta, "log_lik_train": fit.log_lik,
                     "n_obs_train": int(train_mask.sum())})
m2_long = pd.DataFrame(rows)

m = sig_full.merge(m2_long[["iso3", "year", "e_star_trainwindow"]], on=["iso3", "year"], how="inner")
m = m.dropna(subset=["e_star", "e_star_trainwindow"])
both = m[m["year"] <= 2014]
testw = m[m["year"] >= 2015]


def corr(a, b):
    if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


per_country = []
for iso3, gg in m.groupby("iso3"):
    if len(gg) < 20:
        continue
    per_country.append({
        "iso3": iso3,
        "n": int(len(gg)),
        "pearson_r_full": round(corr(gg["e_star"].to_numpy(), gg["e_star_trainwindow"].to_numpy()), 4),
        "rmse": round(float(np.sqrt(np.mean((gg["e_star"] - gg["e_star_trainwindow"]) ** 2))), 4),
        "mean_abs_diff": round(float(np.mean(np.abs(gg["e_star"] - gg["e_star_trainwindow"]))), 4),
        "sign_agreement": round(float(np.mean(np.sign(gg["e_star"]) == np.sign(gg["e_star_trainwindow"]))), 4),
    })
pc = pd.DataFrame(per_country)

feats = pd.read_csv(get_path("features_file")).reset_index(drop=True)
expanded = m2_long[["iso3", "year", "e_star_trainwindow"]]
feats2 = feats.merge(expanded, on=["iso3", "year"], how="left")
ok = feats2["F1_e_star"].notna() & feats2["e_star_trainwindow"].notna()
corr_f1 = float(np.corrcoef(feats2.loc[ok, "F1_e_star"], feats2.loc[ok, "e_star_trainwindow"])[0, 1])

# 异常标记一致率
anom_full = sig_full.set_index(["iso3", "year"])["is_anomaly"]
anom_new = expanded.set_index(["iso3", "year"])["e_star_trainwindow"].apply(
    lambda v: int(abs(v) >= 2) if pd.notna(v) else 0)
common = anom_full.index.intersection(anom_new.index)
anom_agree = float((anom_full.loc[common].astype(int) == anom_new.loc[common].astype(int)).mean())

res_m2 = {
    "finding": "管线的 Kalman 超参在每国全序列（1970-2024，含测试窗）上以 MLE 估计；"
               "一步预测残差本身因果，但超参使用了测试期信息。",
    "n_countries_refit": int(m2_long["iso3"].nunique()),
    "n_obs_train_range": [int(m2_long["n_obs_train"].min()), int(m2_long["n_obs_train"].max())],
    "pooled_pearson_r_all_years": round(corr(m["e_star"], m["e_star_trainwindow"]), 4),
    "pooled_pearson_r_train_window": round(corr(both["e_star"], both["e_star_trainwindow"]), 4),
    "pooled_pearson_r_test_window": round(corr(testw["e_star"], testw["e_star_trainwindow"]), 4),
    "per_country_pearson_r_median": round(float(pc["pearson_r_full"].median()), 4) if len(pc) else None,
    "per_country_pearson_r_min": round(float(pc["pearson_r_full"].min()), 4) if len(pc) else None,
    "per_country_pearson_r_p25": round(float(pc["pearson_r_full"].quantile(0.25)), 4) if len(pc) else None,
    "per_country_n": int(len(pc)),
    "median_rmse": round(float(pc["rmse"].median()), 4) if len(pc) else None,
    "median_sign_agreement": round(float(pc["sign_agreement"].median()), 4) if len(pc) else None,
    "f1_feature_corr_with_trainwindow_variant": round(corr_f1, 4),
    "anomaly_flag_agreement_rate": round(anom_agree, 4),
    "n_diverged_countries_pipeline": int(params_full["diverged"].sum()),
    "n_countries_pipeline": int(len(params_full)),
    "n_obs_pipeline_range": [int(params_full["n_obs"].min()), int(params_full["n_obs"].max())],
    "n_converged_pipeline": int(params_full["converged"].sum()),
}
print(json.dumps(res_m2, ensure_ascii=False, indent=1))
(OUT / "m2_window_sensitivity.json").write_text(json.dumps(res_m2, ensure_ascii=False, indent=2),
                                                encoding="utf-8")
pc.to_csv(OUT / "m2_per_country_comparison.csv", index=False)

# ============ M3: e* 与价格指标的相关性 ============
if "inflation" in panel.columns:
    p3 = panel[["iso3", "year", "inflation"]].merge(sig_full, on=["iso3", "year"], how="inner")
    p3 = p3.dropna(subset=["inflation", "e_star"])

    def within_corr(df, x, y):
        d = df.copy()
        d["xd"] = d[x] - d.groupby("iso3")[x].transform("mean")
        d["yd"] = d[y] - d.groupby("iso3")[y].transform("mean")
        ok = d[["xd", "yd"]].notna().all(axis=1)
        d = d[ok]
        if len(d) < 3 or d["xd"].std() == 0 or d["yd"].std() == 0:
            return float("nan")
        return float(np.corrcoef(d["xd"], d["yd"])[0, 1])

    res_m3 = {
        "n_obs": int(len(p3)),
        "pooled_corr_e_star_inflation": round(corr(p3["e_star"], p3["inflation"]), 4),
        "within_country_corr_e_star_inflation": round(within_corr(p3, "e_star", "inflation"), 4),
        "share_inflation_missing": round(float(panel["inflation"].isna().mean()), 4),
        "interpretation": "若 |相关| 接近 0，说明 e* 并非主要由通胀率共同驱动；但军费为现价美元"
                          "（未平减），故 e* 仍含价格水平共同趋势成分，应解释为'名义增速相对自身趋势的偏离'。",
    }
    f = feats[["iso3", "year", "F1_e_star", "F3_inflation_lag1"]].dropna()
    res_m3["corr_F1_vs_F3_inflation"] = round(corr(f["F1_e_star"], f["F3_inflation_lag1"]), 4)
    print(json.dumps(res_m3, ensure_ascii=False, indent=1))
    (OUT / "m3_price_diagnostic.json").write_text(json.dumps(res_m3, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")

print("DONE: m2_window_sensitivity.json / m3_price_diagnostic.json")
