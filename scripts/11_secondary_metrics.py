# -*- coding: utf-8 -*-
"""N4（pAUC / precision@100 的 bootstrap 95% CI）+ N6（F3 零填充剔除对照）。

规范来源：论文 revision_v3 release_package scripts/compute_n4_n6.py。
golden：pAUC CI 0.010[0,0.072]/0.110[0.044,0.186]/0.112[0.051,0.225]；
        precision@100 0.020[0,0.040]/0.080[0.030,0.130]/0.070[0.020,0.130]；
        N6：90 行（4 阳性）剔除后 KF 0.5379→0.5538、LogReg 0.5497→0.5329。

输出：artifacts/n4_bootstrap_ci.json、artifacts/n6_f3_sensitivity.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, roc_curve

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conflict_ew.baselines import baseline_var  # noqa: E402
from conflict_ew.factors import get_factor_columns  # noqa: E402
from conflict_ew.paths import cfg, get_path  # noqa: E402

OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

cv = cfg()["model"]["cv"]
TS, TE, CS = cv["test_start"], cv["test_end"], cv["cv_start"]
feats = pd.read_csv(get_path("features_file")).reset_index(drop=True)
panel = pd.read_csv(get_path("panel_file"))
oos = pd.read_csv(get_path("oos_file"))
fc = get_factor_columns()

lab = {(r.iso3, int(r.year)): int(r.y_onset) for r in feats.itertuples()}
oos["y"] = pd.to_numeric([lab.get((c, int(y)), None) for c, y in zip(oos["country"], oos["year"])],
                         errors="coerce")
test = oos[oos["fold"] == "test"].dropna(subset=["y"]).copy()
test["y"] = test["y"].astype(int)
y_kf = test["y"].to_numpy()
p_kf = test["p_conflict"].to_numpy()

# ---------- LogReg(RE)（与 baselines.py 同规格：训练窗拟合 + 全行预测） ----------
p2 = panel[["iso3", "year", "mil_expense_log"]].sort_values(["iso3", "year"]).copy()
p2["mil_growth"] = p2.groupby("iso3")["mil_expense_log"].diff()
p2["raw_exp_growth"] = p2.groupby("iso3")["mil_growth"].shift(1)
cols = [c for c in fc if c != "F1_e_star"]
X = feats[["iso3", "year"] + cols].merge(p2[["iso3", "year", "raw_exp_growth"]],
                                         on=["iso3", "year"], how="left")
vals_x = X.drop(columns=["iso3", "year"]).fillna(0.0)
vals_x["raw_exp_growth"] = X["raw_exp_growth"].fillna(0.0).to_numpy()
yv = feats["y_onset"].to_numpy()
tr = ((feats["year"] >= CS) & (feats["year"] < TS)).to_numpy()
spw = (len(yv) - yv.sum()) / max(yv.sum(), 1)
clf = LogisticRegression(C=0.1, max_iter=3000, class_weight={0: 1.0, 1: spw})
clf.fit(vals_x[tr], yv[tr])
fp = feats[["iso3", "year"]].rename(columns={"iso3": "country"}).copy()
fp["p"] = clf.predict_proba(vals_x)[:, 1]
test_lr = test.drop(columns=["p_conflict"]).merge(fp, on=["country", "year"], how="inner")
y_lr = test_lr["y"].to_numpy()
p_lr = test_lr["p"].to_numpy()

# ---------- VAR(2)（baselines.py 规范实现） ----------
bl_va = baseline_var(feats)
lab_va = {(r["iso3"], int(r["year"])): int(r["y_onset"]) for _, r in feats.iterrows()}
y_va = np.array([lab_va[(r["country"], r["year"])] for r in bl_va["test_rows"]])
p_va = np.array([r["p_conflict"] for r in bl_va["test_rows"]])

# ---------- N4: bootstrap CI ----------
def partial_auc(yv_, pv_, max_fpr=0.1):
    fpr, tpr, _ = roc_curve(yv_, pv_)
    m = fpr <= max_fpr
    return float(np.trapezoid(tpr[m], fpr[m]) / max_fpr)


def prec_at_k(yv_, pv_, k=100):
    o = np.argsort(-np.asarray(pv_))[:k]
    return float(np.asarray(yv_)[o].mean())


def boot_ci(fn, yv_, pv_, n_boot=1000, seed=20240101):
    rng = np.random.default_rng(seed)
    n = len(yv_)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(yv_[idx])) < 2:
            continue
        vals.append(fn(yv_[idx], pv_[idx]))
    vals = np.array(vals)
    return {"point": round(float(fn(yv_, pv_)), 4),
            "ci_lower": round(float(np.percentile(vals, 2.5)), 4),
            "ci_upper": round(float(np.percentile(vals, 97.5)), 4),
            "n_boot": int(len(vals))}


n4 = {}
for name, yv_, pv_ in [("KF+MF", y_kf, p_kf), ("LogReg_RE", y_lr, p_lr), ("VAR2", y_va, p_va)]:
    n4[name] = {
        "partial_auc_0_0.1": boot_ci(partial_auc, yv_, pv_),
        "precision_at_100": boot_ci(prec_at_k, yv_, pv_),
    }
    print(name, json.dumps(n4[name]))
(OUT / "n4_bootstrap_ci.json").write_text(json.dumps(n4, indent=2), encoding="utf-8")

# ---------- N6: F3 零填充剔除对照 ----------
te_panel = panel[(panel["year"] >= TS) & (panel["year"] <= TE)][
    ["iso3", "year", "F3_gdp_growth_lag1", "F3_inflation_lag1", "F3_unemployment_lag1"]].rename(
    columns={"iso3": "country"})
te_panel = te_panel.drop_duplicates(subset=["country", "year"])
tt = test.merge(te_panel, on=["country", "year"], how="left")
f3_missing = tt[["F3_gdp_growth_lag1", "F3_inflation_lag1", "F3_unemployment_lag1"]].isna().any(axis=1)
n_missing = int(f3_missing.sum())
pos_missing = int(tt.loc[f3_missing, "y"].sum())

mask_keep = ~f3_missing.values
y_sub = tt.loc[mask_keep, "y"].to_numpy()
p_sub = tt.loc[mask_keep, "p_conflict"].to_numpy()
auc_full = roc_auc_score(y_kf, p_kf)
auc_sub = roc_auc_score(y_sub, p_sub)

tt_lr = test_lr.merge(te_panel, on=["country", "year"], how="left")
miss_lr = tt_lr[["F3_gdp_growth_lag1", "F3_inflation_lag1", "F3_unemployment_lag1"]].isna().any(axis=1)
auc_lr_full = roc_auc_score(y_lr, p_lr)
auc_lr_sub = roc_auc_score(tt_lr.loc[~miss_lr.values, "y"].to_numpy(),
                           tt_lr.loc[~miss_lr.values, "p"].to_numpy())

n6 = {
    "rule": "对照口径 = 从测试窗评估中剔除任一 F3 分量原始缺失的行（零填充前的 NaN 掩码），预测不改、不重训练",
    "n_test_full": int(len(y_kf)),
    "n_imputed_rows": n_missing,
    "imputed_share": round(n_missing / len(y_kf), 4),
    "positives_in_imputed": pos_missing,
    "KF_MF": {"auc_full": round(float(auc_full), 4), "auc_drop_imputed": round(float(auc_sub), 4),
              "diff": round(float(auc_sub - auc_full), 4)},
    "LogReg_RE": {"auc_full": round(float(auc_lr_full), 4), "auc_drop_imputed": round(float(auc_lr_sub), 4),
                  "diff": round(float(auc_lr_sub - auc_lr_full), 4)},
}
print(json.dumps(n6, indent=1))
(OUT / "n6_f3_sensitivity.json").write_text(json.dumps(n6, indent=2), encoding="utf-8")
print("DONE: n4_bootstrap_ci.json, n6_f3_sensitivity.json")
