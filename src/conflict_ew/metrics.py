"""F09 回测评估与基准对比。

- 5 指标：Precision / Recall / AUC(DeLong 95% CI) / Brier 分解 / 预警提前量
- 基准对比：KF+MF（主模型）vs logistic(RE) vs VAR（bootstrap 差异检验 + FDR 校正）
- 消融 A1-A4（与主模型同一 blocked CV 协议，防泄漏）
- 阈值选择仅在训练期验证折完成（Youden's J / Fβ=2，§8.2）
- 输出 metrics.json（对齐论文表 1 数值）
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd
from scipy.stats import rankdata
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

    # ---- 基准对比（论文 Table III 规范实现，见 baselines.py）----
    from .baselines import (baseline_logistic_re, baseline_var, delong_ci,
                            paired_bootstrap, partial_auc)
    bl_re = baseline_logistic_re(features, get_factor_columns())
    bl_var = baseline_var(features)
    metrics["baselines"] = {}
    diffs = []
    for name, b in (("logistic_RE", bl_re), ("VAR2", bl_var)):
        summary = {k: v for k, v in b.items() if k != "test_rows"}
        # 与主模型测试期逐行对齐 → 规范配对 bootstrap（Table III 同式）
        p_map = {(r["country"], r["year"]): r["p_conflict"] for r in b["test_rows"]}
        keys = list(zip(test["country"], test["year"]))
        mask = np.array([k in p_map for k in keys])
        y_al = y_test[mask]
        p_al = np.array([p_map[k] for k, m in zip(keys, mask) if m])
        if mask.sum() >= 20:
            d = paired_bootstrap(y_al, p_test[mask], p_al, n_boot=ecfg["bootstraps"])
            d["baseline"] = name
            d["n_paired"] = int(mask.sum())
        else:
            d = {"baseline": name, "diff": None, "p_value": None,
                 "note": "基准样本不足或训练失败"}
        diffs.append(d)
        metrics["baselines"][name] = summary
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
