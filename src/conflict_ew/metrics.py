"""F09 回测评估与基准对比。

- 5 指标：Precision / Recall / AUC(DeLong 95% CI) / Brier 分解 / 预警提前量
- 基准对比：KF+MF（主模型）vs logistic(RE) vs VAR（bootstrap 差异检验 + FDR 校正）
- 消融 A1-A4（与主模型同一 blocked CV 协议，防泄漏）
- 阈值选择仅在训练期验证折完成（Youden's J / Fβ=2，§8.2）
- 输出 metrics.json（对齐论文表 1 数值）
"""
from __future__ import annotations

import json
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import meta
from .factors import get_factor_columns
from .paths import cfg, get_path
from .xgb import run_blocked_cv


# ---------------------------------------------------------------- DeLong AUC CI
def delong_auc_var(y_true: np.ndarray, scores: np.ndarray) -> float:
    """DeLong 方差（Sun & Xu 快速算法），返回 AUC 的方差估计。"""
    y = np.asarray(y_true, dtype=bool)
    s = np.asarray(scores, dtype=float)
    n1 = int(y.sum())
    n0 = int(len(y) - n1)
    if n1 == 0 or n0 == 0:
        return float("nan")
    r = rankdata(s)
    r1, r0 = r[y], r[~y]
    theta = float((np.sum(r1[:, None] > r0[None, :]) + 0.5 * np.sum(r1[:, None] == r0[None, :])) / (n1 * n0))
    V10 = np.array([(np.sum(r1[i] > r0) + 0.5 * np.sum(r1[i] == r0)) / n0 for i in range(n1)])
    V01 = np.array([(np.sum(r0[j] < r1) + 0.5 * np.sum(r0[j] == r1)) / n1 for j in range(n0)])
    S10 = np.sum((V10 - theta) ** 2) / max(n1 - 1, 1)
    S01 = np.sum((V01 - theta) ** 2) / max(n0 - 1, 1)
    return float(S10 / n1 + S01 / n0)


def auc_ci(y_true, scores, level: float = 0.95) -> dict:
    """AUC + DeLong 95% CI。"""
    from scipy.stats import norm

    auc = float(roc_auc_score(y_true, scores))
    var = delong_auc_var(y_true, scores)
    if not np.isfinite(var) or var <= 0:
        return {"auc": auc, "ci_lower": np.nan, "ci_upper": np.nan, "ci_method": "DeLong"}
    z = norm.ppf(1 - (1 - level) / 2)
    half = z * np.sqrt(var)
    return {"auc": auc, "ci_lower": auc - half, "ci_upper": auc + half, "ci_method": "DeLong"}


# ---------------------------------------------------------------- Brier 分解
def brier_decomposition(y_true: np.ndarray, p: np.ndarray, n_bins: int = 10) -> dict:
    """Brier = Reliability - Resolution + Uncertainty（分箱估计）。"""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(p, dtype=float).clip(0, 1)
    brier = float(np.mean((p - y) ** 2))
    unc = float(np.mean(y) * (1 - np.mean(y)))
    edges = np.linspace(0, 1, n_bins + 1)
    rel = res = 0.0
    N = len(y)
    for k in range(n_bins):
        lo, hi = edges[k], edges[k + 1]
        mask = (p > lo) & (p <= hi) if k > 0 else (p >= lo) & (p <= hi)
        if not mask.any():
            continue
        nk = mask.sum()
        pk, yk = p[mask].mean(), y[mask].mean()
        rel += (nk / N) * (pk - yk) ** 2
        res += (nk / N) * (yk - np.mean(y)) ** 2
    return {"brier": brier, "reliability": float(rel), "resolution": float(res),
            "uncertainty": float(unc)}


# ---------------------------------------------------------------- 预警提前量
def lead_time_analysis(oos_test: pd.DataFrame, panel: pd.DataFrame,
                       threshold: float = 0.7) -> dict:
    """预警提前量：首次 p≥threshold → COW onset 的平均月数（测试期）。"""
    test = oos_test[oos_test["fold"] == "test"].copy()
    onsets = panel[(panel["y_onset"] == 1) &
                   (panel["year"] >= cfg()["model"]["cv"]["test_start"]) &
                   (panel["year"] <= cfg()["model"]["cv"]["test_end"])]
    leads = []
    covered = 0
    for _, r in onsets.iterrows():
        iso3, y0 = r["iso3"], int(r["year"])
        cnt = test[(test["country"] == iso3) & (test["year"] <= y0)]
        if len(cnt) == 0:
            continue
        alarms = cnt[cnt["p_conflict"] >= threshold]
        if len(alarms) == 0:
            continue
        first_alarm = int(alarms["year"].min())
        leads.append((y0 - first_alarm) * 12)
        covered += 1
    return {"mean_lead_months": float(np.mean(leads)) if leads else None,
            "n_onsets_covered": covered, "n_onsets_total": len(onsets),
            "coverage_rate": covered / len(onsets) if len(onsets) else None,
            "threshold": threshold}


# ---------------------------------------------------------------- 基准模型
def baseline_logistic_re(features: pd.DataFrame, feat_cols: list[str]) -> tuple[list[dict], float]:
    """logistic(RE) 基准：面板 logit + country 固定效应（随机效应的近似，已标注）。

    严格防泄漏：只在训练期拟合，测试期预测；返回 (rows, test_auc)。
    """
    cv = cfg()["model"]["cv"]
    df = features.copy()
    X = df[feat_cols].fillna(0.0)
    # country 固定效应
    country_dummies = pd.get_dummies(df["iso3"], prefix="c")
    Xf = pd.concat([X.reset_index(drop=True), country_dummies.reset_index(drop=True)], axis=1)
    y = df["y_onset"].to_numpy()
    spw = (len(y) - y.sum()) / max(y.sum(), 1)

    train_mask = (df["year"] >= cv["cv_start"]) & (df["year"] < cv["test_start"])
    test_mask = (df["year"] >= cv["test_start"]) & (df["year"] <= cv["test_end"])
    clf = LogisticRegression(C=0.1, max_iter=2000, class_weight={0: 1.0, 1: spw})
    clf.fit(Xf[train_mask], y[train_mask])
    p_test = clf.predict_proba(Xf[test_mask])[:, 1]
    rows = [{"country": c, "year": int(yy), "p_conflict": float(p),
             "model": "logistic_RE"} for c, yy, p in
            zip(df[test_mask]["iso3"], df[test_mask]["year"], p_test)]
    auc = float(roc_auc_score(y[test_mask], p_test)) if len(np.unique(y[test_mask])) > 1 else float("nan")
    return rows, auc


def baseline_var(features: pd.DataFrame) -> tuple[list[dict], float]:
    """VAR(1) 基准：对每国 [mil_growth, gdp_growth] 拟合 pooled VAR(1)，
    一步预测军费增速 z 分 → 训练期校准的 logistic 链接转概率（已标注简化）。"""
    cv = cfg()["model"]["cv"]
    df = features.sort_values(["iso3", "year"]).copy()
    # 军费/经济序列来自面板（features 不含原始值，避免重复存储）
    panel = pd.read_csv(get_path("panel_file"))
    df = df.merge(panel[["iso3", "year", "mil_expense_log", "gdp_growth"]],
                  on=["iso3", "year"], how="left")
    df["mil_growth"] = df.groupby("iso3")["mil_expense_log"].diff()
    df["lag_mil"] = df.groupby("iso3")["mil_growth"].shift(1)
    df["lag_gdp"] = df.groupby("iso3")["gdp_growth"].shift(1)

    # 训练期 pooled VAR(1)：mil_growth_t = a0 + a1*lag_mil + a2*lag_gdp
    tr = df[(df["year"] >= cv["cv_start"]) & (df["year"] < cv["test_start"])].dropna(
        subset=["mil_growth", "lag_mil", "lag_gdp"])
    A = np.column_stack([np.ones(len(tr)), tr["lag_mil"], tr["lag_gdp"]])
    coef, *_ = np.linalg.lstsq(A, tr["mil_growth"], rcond=None)
    tr["z_pred"] = (tr["mil_growth"] - (A @ coef)) / max(tr["mil_growth"].std(), 1e-9)

    # 训练期校准：sigmoid(a + b*z)
    zz = tr["z_pred"].to_numpy()
    yy = tr["y_onset"].to_numpy()
    design = np.column_stack([np.ones(len(zz)), zz])
    bc, *_ = np.linalg.lstsq(design, np.clip(yy, 0.05, 0.95), rcond=None)

    te = df[(df["year"] >= cv["test_start"]) & (df["year"] <= cv["test_end"])].dropna(
        subset=["mil_growth", "lag_mil", "lag_gdp"])
    B = np.column_stack([np.ones(len(te)), te["lag_mil"], te["lag_gdp"]])
    z_test = (te["mil_growth"] - (B @ coef)) / max(tr["mil_growth"].std(), 1e-9)
    p_test = 1 / (1 + np.exp(-(bc[0] + bc[1] * z_test)))
    rows = [{"country": c, "year": int(yy), "p_conflict": float(p), "model": "VAR"}
            for c, yy, p in zip(te["iso3"], te["year"], p_test)]
    auc = float(roc_auc_score(te["y_onset"], p_test)) if len(np.unique(te["y_onset"])) > 1 else float("nan")
    return rows, auc


# ---------------------------------------------------------------- 消融
ABLATION_SPECS = {
    "A1_no_mil_signal": {"label": "消融 A1：去掉 F1 军费异常信号", "drop": ["F1_e_star"]},
    "A2_no_arms_race": {"label": "消融 A2：去掉 F2 军备竞赛", "drop": ["F2_arms_race"]},
    "A3_no_econ_pol": {"label": "消融 A3：去掉 F3 经济压力 + F4 政治不稳",
                       "drop": [c for c in get_factor_columns() if c.startswith(("F3_", "F4_"))]},
    "A4_only_mil_signal": {"label": "消融 A4：仅保留 F1 军费异常信号",
                           "keep_only": ["F1_e_star"]},
}


def _subset_cols(spec: dict) -> list[str]:
    all_cols = get_factor_columns()
    if "keep_only" in spec:
        return [c for c in all_cols if c in spec["keep_only"]]
    return [c for c in all_cols if c not in spec["drop"]]


def run_ablation(features: pd.DataFrame) -> list[dict]:
    """消融 A1-A4：与主模型同一 blocked CV 协议，比较测试期 AUC。"""
    outs = []
    for key, spec in ABLATION_SPECS.items():
        cols = _subset_cols(spec)
        try:
            run = run_blocked_cv(features, feat_cols=cols)
            test_rows = pd.DataFrame(run.test_rows)
            feats = features[(features["year"] >= cfg()["model"]["cv"]["test_start"]) &
                             (features["year"] <= cfg()["model"]["cv"]["test_end"])]
            auc = float(roc_auc_score(feats["y_onset"], test_rows["p_conflict"])) \
                if len(np.unique(feats["y_onset"])) > 1 else float("nan")
            outs.append({"ablation": key, "label": spec["label"], "drop": spec.get("drop", []),
                         "keep_only": spec.get("keep_only", []), "test_auc": auc,
                         "n_feats": len(cols)})
            meta.log_stage("ablation", ablation=key, test_auc=round(auc, 4) if auc == auc else None)
        except Exception as e:  # noqa: BLE001
            outs.append({"ablation": key, "label": spec["label"], "test_auc": None, "error": str(e)})
            meta.log_stage("ablation_failed", ablation=key, error=str(e))
    return outs


# ---------------------------------------------------------------- FDR / bootstrap
def bh_fdr(pvals: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg FDR 校正，返回显著掩码。"""
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    rank = np.empty(m)
    rank[order] = np.arange(1, m + 1)
    threshold = alpha * rank / m
    sig = p <= threshold
    return sig


def bootstrap_auc_diff(y_true, p_a, p_b, n_boot: int = 1000, seed: int = 20240101) -> dict:
    """bootstrap 差异检验（AUC_a - AUC_b 的 95% CI 与双侧 p 值）。"""
    rng = np.random.default_rng(seed)
    y = np.asarray(y_true)
    n = len(y)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    if len(pos_idx) < 2 or len(neg_idx) < 2:
        return {"diff": None, "ci_lower": None, "ci_upper": None, "p_value": None, "method": "bootstrap"}
    diffs = []
    for _ in range(n_boot):
        pi = rng.choice(pos_idx, size=len(pos_idx), replace=True)
        ni = rng.choice(neg_idx, size=len(neg_idx), replace=True)
        idx = np.concatenate([pi, ni])
        a = roc_auc_score(y[idx], p_a[idx])
        b = roc_auc_score(y[idx], p_b[idx])
        diffs.append(a - b)
    diffs = np.array(diffs)
    diff = float(np.mean(diffs))
    ci = np.percentile(diffs, [2.5, 97.5])
    p_value = float(2 * min((diffs > 0).mean(), (diffs < 0).mean()))
    return {"diff": diff, "ci_lower": float(ci[0]), "ci_upper": float(ci[1]),
            "p_value": p_value, "method": "bootstrap-1000"}


# ---------------------------------------------------------------- 主入口
def evaluate(features: pd.DataFrame, oos: pd.DataFrame, panel: pd.DataFrame,
             fold_results=None) -> dict:
    """F09 主入口：计算全部指标并输出 metrics.json。

    :param features: 特征表（含 y_onset）
    :param oos: oos_predictions（val 折 + test）
    :param panel: 面板（含 y_onset，用于提前量）
    :param fold_results: 主模型 CV 折结果（可选）
    """
    ecfg = cfg()["eval"]
    test = oos[oos["fold"] == "test"].copy()
    test_feats = features[(features["year"] >= cfg()["model"]["cv"]["test_start"]) &
                          (features["year"] <= cfg()["model"]["cv"]["test_end"])]
    y_test = test_feats.set_index(["iso3", "year"]).loc[
        list(zip(test["country"], test["year"]))]["y_onset"].to_numpy()
    p_test = test["p_conflict"].to_numpy()

    metrics = {"model": "KF+MF (Kalman + Multi-Factor XGBoost)",
               "model_version": cfg()["project"]["version"],
               "test_period": f"{cfg()['model']['cv']['test_start']}-{cfg()['model']['cv']['test_end']}"}

    # 阈值（用 config 默认 0.7/0.5；训练期建议阈值单独输出）
    hi, mid = ecfg["thresholds"]["high"], ecfg["thresholds"]["mid"]
    pred_pos = p_test >= hi
    tp = float((pred_pos & (y_test == 1)).sum())
    fp = float((pred_pos & (y_test == 0)).sum())
    fn = float((~pred_pos & (y_test == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    metrics["precision"] = {"value": precision, "threshold": hi}
    metrics["recall"] = {"value": recall, "threshold": hi}
    metrics["auc"] = auc_ci(y_test, p_test)
    metrics["brier"] = brier_decomposition(y_test, p_test)
    metrics["lead_time"] = lead_time_analysis(oos, panel, threshold=hi)
    metrics["n_test_rows"] = int(len(y_test))
    metrics["n_test_positive"] = int(y_test.sum())
    metrics["thresholds_used"] = {"high": hi, "mid": mid}

    # ---- 训练期建议阈值（Youden's J / Fβ=2，只在验证折上选，防泄漏）----
    val = oos[oos["fold"].str.startswith("val_fold")].copy()
    suggested_t = None
    if len(val):
        val_feats = features.set_index(["iso3", "year"])
        y_val = np.array([int(val_feats.loc[(r["country"], r["year"]), "y_onset"]) for _, r in val.iterrows()])
        p_val = val["p_conflict"].to_numpy()
        beta = ecfg.get("youden_beta", 2.0)
        cands = np.quantile(p_val, np.linspace(0.2, 0.9, 50))
        best_t, best_f = None, -1.0
        for t in cands:
            pred = p_val >= t
            r_ = (pred & (y_val == 1)).sum() / max((y_val == 1).sum(), 1)
            pr = (pred & (y_val == 1)).sum() / max(pred.sum(), 1)
            if r_ + pr == 0:
                continue
            fbeta = (1 + beta ** 2) * pr * r_ / (beta ** 2 * pr + r_)
            if fbeta > best_f:
                best_f, best_t = fbeta, float(t)
        suggested_t = best_t
        metrics["suggested_threshold"] = {"value": best_t, "fbeta": float(best_f), "beta": beta,
                                          "note": "训练期验证折确定（防泄漏）；运行阈值取 config.yaml 默认 0.7/0.5"}
        # @建议阈值口径的 Recall/Precision（与 @0.7 口径并列，展示模型判别能力）
        if best_t is not None:
            pred_s = p_test >= best_t
            tp_s = float((pred_s & (y_test == 1)).sum())
            fp_s = float((pred_s & (y_test == 0)).sum())
            fn_s = float((~pred_s & (y_test == 1)).sum())
            metrics["recall_at_suggested"] = {"value": tp_s / max(tp_s + fn_s, 1), "threshold": best_t}
            metrics["precision_at_suggested"] = {"value": tp_s / max(tp_s + fp_s, 1), "threshold": best_t}
            metrics["lead_time_at_suggested"] = lead_time_analysis(oos, panel, threshold=best_t)

    # ---- 基准对比 ----
    baselines = {}
    logit_rows, logit_auc = baseline_logistic_re(features, get_factor_columns())
    var_rows, var_auc = baseline_var(features)
    baselines["logistic_RE"] = {"auc": logit_auc if logit_auc == logit_auc else None,
                                "label": "logistic(RE) 基准（country 固定效应近似）",
                                "model": "logistic_RE"}
    baselines["VAR"] = {"auc": var_auc if var_auc == var_auc else None,
                        "label": "VAR(1) 基准（军费增速/GDP 增速，简化实现）",
                        "model": "VAR"}
    # 差异检验 + FDR 校正（与主模型对比）
    diffs = []
    for name, rows, auc_b in (("logistic_RE", logit_rows, logit_auc), ("VAR", var_rows, var_auc)):
        if not np.isfinite(auc_b) or len(rows) == 0:
            diffs.append({"baseline": name, "diff": None, "p_value": None,
                          "note": "基准训练失败（降级）"})
            continue
        p_b = np.array([r["p_conflict"] for r in rows])
        # 对齐 y（同一测试期样本）
        key_b = {(r["country"], r["year"]): i for i, r in enumerate(rows)}
        y_al = np.array([y_test[list(zip(test["country"], test["year"])).index((c, yy))]
                         if (c, yy) in key_b else np.nan
                         for c, yy in zip(test["country"], test["year"])])
        p_al = np.array([p_b[key_b[(c, yy)]] if (c, yy) in key_b else np.nan
                         for c, yy in zip(test["country"], test["year"])])
        mask = np.isfinite(y_al) & np.isfinite(p_al)
        if mask.sum() < 20:
            diffs.append({"baseline": name, "diff": None, "p_value": None, "note": "样本不足"})
            continue
        d = bootstrap_auc_diff(y_al[mask], p_test[mask], p_al[mask],
                               n_boot=ecfg["bootstraps"])
        d["baseline"] = name
        diffs.append(d)
    # FDR
    pvals = np.array([d.get("p_value", 1.0) if d.get("p_value") is not None else 1.0 for d in diffs])
    sig = bh_fdr(pvals, alpha=ecfg["fdr_alpha"])
    for i, d in enumerate(diffs):
        d["fdr_significant"] = bool(sig[i])
    metrics["baseline_comparison"] = diffs
    metrics["fdr_alpha"] = ecfg["fdr_alpha"]
    metrics["bootstrap_n"] = ecfg["bootstraps"]

    # ---- 消融 ----
    metrics["ablation"] = run_ablation(features)

    # ---- CV 折 ----
    if fold_results:
        metrics["cv_folds"] = [fr.__dict__ for fr in fold_results]

    with open(get_path("metrics_file"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2, default=str)
    meta.log_stage("evaluate", test_auc=metrics["auc"]["auc"], recall=recall, precision=precision,
                   brier=metrics["brier"]["brier"])
    return metrics
