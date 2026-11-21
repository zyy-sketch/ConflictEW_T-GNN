"""基准模型（论文 Table III 规范实现的 1:1 移植）。

权威来源：论文 revision_v3 release_package `scripts/table3_unified.py`
（其产物 `artifacts/table3_unified.json` 与稿件 TABLE III 逐值一致）。
本模块移植其 LogReg(RE) 与 VAR(2) 两个基线块，供管道复现论文数值。

工作点规则（TABLE III 统一口径）：Fβ=2 最优的**训练期预测分位数**
（网格 np.arange(0.005, 0.999, 0.005)）。注意：主模型 KF+MF 的 τ 来自
oos 验证折；两个基线的 τ 来自各自训练期 in-sample 预测——这是规范
实现的行为，论文 §V-B 文字如与此次口径有出入，以论文回写统一（R-11）。

LogReg (RE)：稀有事件类加权（scale_pos_weight，按全特征表计）、无国家
固定效应；协变量 = 主模型经济政治指标（F2–F6）+ **滞后一年**的未滤波
log 军费同比增速（替代滤波残差 F1）；C=0.1。

VAR(2)：pooled 二阶自回归（[军费增速, GDP 增速] 的两阶滞后），训练窗
OLS；训练期链接 = 对 clip(y,0.02,0.98) 的 OLS 拟合后过 sigmoid；测试期
z 用预测值（训练期 z 用残差）——均为规范实现的原始行为，不做"修正"。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from .metrics import lead_time_analysis
from .paths import cfg, get_path

_TAU_GRID = np.arange(0.005, 0.999, 0.005)  # 规范分位网格


# ---------------------------------------------------------------- 规范工具
def fbeta_confusion(y: np.ndarray, p: np.ndarray, t: float) -> dict:
    """规范 Fβ=2 混淆矩阵（β=2：f = 5pr/(4p+r)）。"""
    pr = p >= t
    tp = int((pr & (y == 1)).sum())
    fp = int((pr & (y == 0)).sum())
    fn = int(((~pr) & (y == 1)).sum())
    tn = int(((~pr) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f = 5 * prec * rec / (4 * prec + rec) if (prec + rec) else 0.0
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "flagged": tp + fp,
            "precision": prec, "recall": rec, "fbeta2": f}


def best_quantile(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    """Fβ=2 最优预测分位；返回 (阈值, fβ2, 分位 q)。"""
    best_t, best_f, best_q = None, -1.0, None
    for q in _TAU_GRID:
        t = float(np.quantile(p, q))
        m = fbeta_confusion(y, p, t)
        if m["fbeta2"] > best_f:
            best_t, best_f, best_q = t, m["fbeta2"], float(q)
    return best_t, best_f, best_q


def delong_ci(y: np.ndarray, p: np.ndarray, alpha: float = 0.05) -> tuple[float, float, float]:
    """规范 DeLong AUC 区间（Table III 同式）。"""
    y = np.asarray(y)
    p = np.asarray(p)
    n1 = int(y.sum())
    n0 = len(y) - n1
    o = np.argsort(p)
    ps, ys = p[o], y[o]
    nz, nz0 = ps[ys == 1], ps[ys == 0]
    gt = (nz[:, None] > nz0[None, :]).astype(float)
    eq = (nz[:, None] == nz0[None, :]).astype(float)
    v10 = (gt + 0.5 * eq).mean(axis=1)
    v01 = (gt + 0.5 * eq).mean(axis=0)
    a = v10.mean()
    se = float(np.sqrt(v10.var(ddof=1) / n1 + v01.var(ddof=1) / n0))
    from scipy.stats import norm
    z = norm.ppf(1 - alpha / 2)
    return float(a), float(a - z * se), float(a + z * se)


def paired_bootstrap(y: np.ndarray, p1: np.ndarray, p2: np.ndarray,
                     n_boot: int = 1000, seed: int = 20240101) -> dict:
    """规范配对 bootstrap（整索引重采样，Table III 同式）。"""
    rng = np.random.default_rng(seed)
    n = len(y)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        if len(np.unique(y[idx])) < 2:
            continue
        diffs.append(roc_auc_score(y[idx], p1[idx]) - roc_auc_score(y[idx], p2[idx]))
    diffs = np.array(diffs)
    return {"diff": float(roc_auc_score(y, p1) - roc_auc_score(y, p2)),
            "ci_lower": float(np.percentile(diffs, 2.5)),
            "ci_upper": float(np.percentile(diffs, 97.5)),
            "p_value": float(2 * min((diffs <= 0).mean(), (diffs >= 0).mean()))}


def partial_auc(y: np.ndarray, p: np.ndarray, max_fpr: float = 0.1) -> float:
    """pAUC(0–max_fpr)（归一化，Table III 同式）。"""
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, p)
    m = fpr <= max_fpr
    return float(np.trapezoid(tpr[m], fpr[m]) / max_fpr)


# ---------------------------------------------------------------- 公共数据
def _panel_growth() -> pd.DataFrame:
    panel = pd.read_csv(get_path("panel_file"))
    p2 = panel[["iso3", "year", "mil_expense_log", "gdp_growth"]].sort_values(
        ["iso3", "year"]).copy()
    p2["mil_growth"] = p2.groupby("iso3")["mil_expense_log"].diff()
    return p2


def _pack(name: str, label: str, y_te: np.ndarray, p_te: np.ndarray,
          tau: float, q: float, cm: dict, test_df: pd.DataFrame,
          panel: pd.DataFrame, p_train: np.ndarray | None = None,
          y_train: np.ndarray | None = None) -> dict:
    _, lo, hi = delong_ci(y_te, p_te)
    lt = lead_time_analysis(test_df, panel, threshold=tau)
    out = {"model": name, "label": label,
            "n": int(len(y_te)), "n_test_positive": int(y_te.sum()),
            "auc": round(float(roc_auc_score(y_te, p_te)), 4),
            "auc_ci_delong": [round(lo, 4), round(hi, 4)],
            "tau": round(float(tau), 6), "tau_quantile": round(q, 4) if q is not None else None,
            "fbeta2_at_tau": round(cm["fbeta2"], 4),
            "TP": cm["TP"], "FP": cm["FP"], "FN": cm["FN"], "TN": cm["TN"],
            "flagged": cm["flagged"],
            "precision_at_tau": round(cm["precision"], 4),
            "recall_at_tau": round(cm["recall"], 4),
            "brier_raw": round(float(np.mean((y_te - p_te) ** 2)), 4),
            "partial_auc_0_0.1": round(partial_auc(y_te, p_te), 4),
            "lead_time_months": round(lt["mean_lead_months"], 1) if lt["mean_lead_months"] else None,
            "onsets_covered": f"{lt['n_onsets_covered']}/{lt['n_onsets_total']}",
            "test_rows": [{"country": c, "year": int(yy), "p_conflict": float(pp),
                           "fold": "test", "model": name}
                          for c, yy, pp in zip(test_df["country"], test_df["year"], p_te)]}
    if p_train is not None and y_train is not None:
        out["train_preds"] = {"p": [float(v) for v in p_train],
                              "y": [int(v) for v in y_train],
                              "note": "训练窗 in-sample 分数（基线自身校准/阈值口径）"}
    return out

# ---------------------------------------------------------------- (1) LogReg(RE)
def baseline_logistic_re(features: pd.DataFrame, feat_cols: list[str], panel: pd.DataFrame) -> dict:
    """规范 LogReg(RE)：类加权 + 无固定效应 + lag1 原始军费增速替代 F1。"""
    cv = cfg()["model"]["cv"]
    cs, ts, te = cv["cv_start"], cv["test_start"], cv["test_end"]
    p2 = _panel_growth()
    p2["raw_exp_growth"] = p2.groupby("iso3")["mil_growth"].shift(1)
    cols = [c for c in feat_cols if c != "F1_e_star"]
    X = features[["iso3", "year"] + cols].merge(
        p2[["iso3", "year", "raw_exp_growth"]], on=["iso3", "year"], how="left")
    vals = X.drop(columns=["iso3", "year"]).fillna(0.0).copy()
    vals["raw_exp_growth"] = X["raw_exp_growth"].fillna(0.0).to_numpy()
    yv = features["y_onset"].to_numpy()
    tr = ((features["year"] >= cs) & (features["year"] < ts)).to_numpy()
    te_m = ((features["year"] >= ts) & (features["year"] <= te)).to_numpy()
    spw = (len(yv) - yv.sum()) / max(yv.sum(), 1)  # 规范：按全特征表计
    clf = LogisticRegression(C=0.1, max_iter=3000, class_weight={0: 1.0, 1: spw})
    clf.fit(vals[tr], yv[tr])
    p_lr_tr = clf.predict_proba(vals[tr])[:, 1]
    p_lr_te = clf.predict_proba(vals[te_m])[:, 1]
    tau, _f, q = best_quantile(yv[tr], p_lr_tr)
    test_df = pd.DataFrame({"country": features.loc[te_m, "iso3"].to_numpy(),
                            "year": features.loc[te_m, "year"].to_numpy(),
                            "p_conflict": p_lr_te, "fold": "test"})
    y_te = yv[te_m]
    cm = fbeta_confusion(y_te, p_lr_te, tau)
    return _pack("logistic_RE",
                 "LogReg (RE): rare-events class-weighted logistic (no fixed effects; "
                 "lagged unfiltered expenditure growth replaces F1)", y_te, p_lr_te,
                 tau, q, cm, test_df, panel, p_train=p_lr_tr, y_train=yv[tr])


# ---------------------------------------------------------------- (3) Persistence (naive)
def baseline_persistence(features: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """naive persistence 基线（冲突预测文献标准对照）。

    score = 上一年的 active（冲突持续即报警）；无训练、无阈值选择：
    二值分数下 score >= 1 即报警（tau=1.0，tau_quantile=None）。
    """
    cv = cfg()["model"]["cv"]
    ts, te = cv["test_start"], cv["test_end"]
    p2 = panel[["iso3", "year", "active"]].sort_values(["iso3", "year"]).copy()
    p2["active_lag1"] = p2.groupby("iso3")["active"].shift(1)
    feats = features[["iso3", "year", "y_onset"]].merge(
        p2[["iso3", "year", "active_lag1"]], on=["iso3", "year"], how="left")
    te_df = feats[(feats["year"] >= ts) & (feats["year"] <= te)].dropna(subset=["active_lag1"]).copy()
    y_te = te_df["y_onset"].to_numpy().astype(int)
    p_te = te_df["active_lag1"].to_numpy().astype(float)
    tau, q = 1.0, None
    cm = fbeta_confusion(y_te, p_te, tau)
    test_df = pd.DataFrame({"country": te_df["iso3"].to_numpy(),
                            "year": te_df["year"].to_numpy().astype(int),
                            "p_conflict": p_te, "fold": "test"})
    return _pack("persistence",
                 "Persistence (naive): flag countries that were active in conflict in the "
                 "previous year; no training, binary score",
                 y_te, p_te, tau, q, cm, test_df, panel)
# ---------------------------------------------------------------- (2) VAR(2)
def baseline_var(features: pd.DataFrame, panel: pd.DataFrame) -> dict:
    """规范 pooled VAR(2)：OLS 两方程 + OLS-sigmoid 链接（Table III 同式）。"""
    cv = cfg()["model"]["cv"]
    cs, ts, te = cv["cv_start"], cv["test_start"], cv["test_end"]
    dfv = _panel_growth().merge(features[["iso3", "year", "y_onset"]],
                                on=["iso3", "year"], how="left")
    gg = dfv.groupby("iso3")
    dfv["lag1_mil"] = gg["mil_growth"].shift(1)
    dfv["lag1_gdp"] = gg["gdp_growth"].shift(1)
    dfv["lag2_mil"] = gg["mil_growth"].shift(2)
    dfv["lag2_gdp"] = gg["gdp_growth"].shift(2)

    trv = dfv[(dfv["year"] >= cs) & (dfv["year"] < ts)].dropna(
        subset=["mil_growth", "lag1_mil", "lag1_gdp", "lag2_mil", "lag2_gdp"]).copy()
    A = np.column_stack([np.ones(len(trv)), trv["lag1_mil"], trv["lag1_gdp"],
                         trv["lag2_mil"], trv["lag2_gdp"]])
    coef, *_ = np.linalg.lstsq(A, trv["mil_growth"], rcond=None)
    sd = float(trv["mil_growth"].std())
    trv["z_pred"] = (trv["mil_growth"].to_numpy() - A @ coef) / sd
    zz = trv["z_pred"].to_numpy()
    yyv = trv["y_onset"].to_numpy()
    D = np.column_stack([np.ones(len(zz)), zz])
    bc, *_ = np.linalg.lstsq(D, np.clip(yyv, 0.02, 0.98), rcond=None)
    trv["p_hat"] = 1 / (1 + np.exp(-(bc[0] + bc[1] * trv["z_pred"])))
    tau, _f, q = best_quantile(yyv, trv["p_hat"].to_numpy())

    rows = []
    te_df = dfv[(dfv["year"] >= ts) & (dfv["year"] <= te)]
    for iso3, grp in te_df.groupby("iso3"):
        for _, r in grp.sort_values("year").iterrows():
            if pd.isna(r["lag2_mil"]):
                continue
            x = np.array([1.0, r["lag1_mil"], r["lag1_gdp"], r["lag2_mil"], r["lag2_gdp"]])
            z = float(x @ coef) / sd
            rows.append({"country": iso3, "year": int(r["year"]),
                         "p_conflict": float(1 / (1 + np.exp(-(bc[0] + bc[1] * z)))),
                         "fold": "test", "model": "VAR2"})
    var_df = pd.DataFrame(rows)
    lab = {(r["iso3"], int(r["year"])): int(r["y_onset"]) for _, r in features.iterrows()}
    var_df["y"] = [lab.get((r["country"], r["year"]), np.nan) for _, r in var_df.iterrows()]
    var_df = var_df.dropna(subset=["y", "p_conflict"])  # 规范：lag 缺失产生的 NaN p 一并剔除
    y_te = var_df["y"].to_numpy().astype(int)
    p_te = var_df["p_conflict"].to_numpy()
    cm = fbeta_confusion(y_te, p_te, tau)
    return _pack("VAR2",
                 "VAR(2): pooled two-lag autoregression on [mil growth, GDP growth], "
                 "OLS + logistic link calibrated on the training window",
                 y_te, p_te, tau, q, cm, var_df, panel,
                 p_train=trv["p_hat"].to_numpy(), y_train=yyv)
