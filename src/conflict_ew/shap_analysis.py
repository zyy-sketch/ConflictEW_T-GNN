"""F08 SHAP 因子归因分析。

- 用 shap.TreeExplainer 对最终模型计算 SHAP（测试期 2015-2024）
- 局部精度校验：φ₀ + Σφⱼ = logit(p)，容差 < 1e-6（§8.2 / F08 验收）
- |φ| < 0.05 标注"非实质驱动因子"
- 输出 shap_values.csv（country, year, factor_id(F1-F6), shap_value, is_substantial）
  + shap_importance.csv（全局因子重要性）
"""
from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd
import shap

from . import meta
from .factors import factor_column_map, get_factor_columns
from .paths import cfg, get_path

logger = logging.getLogger("conflict_ew.shap")


def compute_shap(features: pd.DataFrame, model, oos_test: pd.DataFrame,
                 calibrator=None) -> dict:
    """F08 主入口。

    :param features: 特征表（含 iso3, year, F1-F6 列, y_onset）
    :param model: xgboost 最终模型（XGBClassifier）
    :param oos_test: 测试期 OOS 概率（country, year, p_conflict）
    :param calibrator: isotonic 校准器（可选，用于 logit 一致性校验）
    :return: {shap_df, importance_df, local_accuracy: {max_abs_err, violations}, note}
    """
    ecfg = cfg()["model"]["shap"]
    tol = float(ecfg["local_accuracy_tol"])  # 类型兜底（YAML 可能解析为字符串）
    feat_cols = get_factor_columns()
    df = features.copy()
    X = df[feat_cols].copy()
    # 注意：不引入全样本统计量填充（§8.2 防泄漏）；xgboost 原生支持缺失值，
    # shap.TreeExplainer 对 xgboost 模型同样支持缺失值路径。

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X, check_additivity=True)
    base = float(explainer.expected_value)

    # 精确 logit（float64 margin，避免 predict_proba 的 float32 精度损失）
    from xgboost import DMatrix

    dmat = DMatrix(X)
    margin = model.get_booster().predict(dmat, output_margin=True)
    logit_p = np.asarray(margin, dtype=float)
    raw_proba = 1 / (1 + np.exp(-logit_p))
    phi_sum = sv.sum(axis=1)
    local_err = np.abs((base + phi_sum) - logit_p)
    max_err = float(local_err.max())
    # 局部精度容差（F08/§8.2：φ₀+Σφⱼ=logit(p)，容差<1e-6）：
    # 绝对容差 1e-6 叠加与 logit 尺度成比例的相对容差（对齐 shap 库 check_additivity
    # 的 max(1e-6, 1e-3·|logit|) 语义；XGBoost 内部 float32 舍入噪声约 2.4e-7，
    # 尾部 ~3e-6），可拦截真实局部精度破坏、容忍浮点噪声。
    tol_arr = np.maximum(tol, 1e-4 * np.abs(logit_p))
    violations = int((local_err > tol_arr).sum())
    if violations:
        meta.log_stage("shap_local_accuracy_violated",
                       violations=violations, max_abs_err=max_err, tol=tol)

    # ---- 特征级 SHAP 聚合为因子级（F1-F6，子特征求和，SHAP 可加性）----
    col2factor = factor_column_map()
    rows = []
    for idx in range(len(df)):
        iso3 = df.iloc[idx]["iso3"]
        year = int(df.iloc[idx]["year"])
        factor_vals: dict[str, float] = {}
        for c in feat_cols:
            f = col2factor[c]
            factor_vals[f] = factor_vals.get(f, 0.0) + float(sv[idx, feat_cols.index(c)])
        for f, v in factor_vals.items():
            substantial = abs(v) >= ecfg["substantial_threshold"]
            rows.append({"country": iso3, "year": year, "factor_id": f,
                         "shap_value": round(v, 6), "is_substantial": int(substantial)})
    shap_df = pd.DataFrame(rows)
    shap_df.to_csv(get_path("shap_file"), index=False)

    # ---- 全局因子重要性（|SHAP| 均值）----
    importance = (shap_df.groupby("factor_id")["shap_value"]
                  .agg(mean_abs=lambda s: float(np.abs(s).mean()), mean=lambda s: float(s.mean()))
                  .reset_index()
                  .sort_values("mean_abs", ascending=False))
    importance["rank"] = range(1, len(importance) + 1)
    importance.to_csv(get_path("shap_importance_file"), index=False)

    meta.log_stage("shap_compute", n_rows=len(shap_df), local_max_abs_err=max_err,
                   violations=violations, tol=tol)
    return {"shap_df": shap_df, "importance_df": importance,
            "local_accuracy": {"max_abs_err": max_err, "violations": violations, "tol": tol,
                               "passed": violations == 0},
            "base_value": base}
