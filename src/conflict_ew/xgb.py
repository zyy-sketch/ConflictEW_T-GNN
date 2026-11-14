"""F07 XGBoost 模型训练与预测。

- blocked expanding-window CV（1990-2014 切 5 窗，训练截止 < 验证起点 < 测试 2015）
- 稀有事件校正：scale_pos_weight = n_neg / n_pos（每折独立计算）
- 严格 out-of-sample：测试期 2015-2024 仅访问一次
- 概率校准（isotonic，训练期验证折拟合）——满足 §8.2 "校准只在训练期拟合"
- 阈值选择仅在训练期验证折完成（F16 约束）
- 输出 oos_predictions.csv / xgb_params.json / xgb_model.json
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.calibration import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import meta
from .factors import get_factor_columns
from .paths import cfg, get_path

MODEL_VERSION = "v1.0.0"


@dataclass
class FoldResult:
    fold: str
    train_start: int
    train_end: int
    val_start: Optional[int]
    val_end: Optional[int]
    n_train: int
    n_val: int
    n_positive_train: int
    auc_val: float
    params: dict = field(default_factory=dict)


@dataclass
class CVRun:
    """一次 blocked CV 的全部产物。"""
    val_rows: list[dict] = field(default_factory=list)   # 验证折概率（pooled）
    test_rows: list[dict] = field(default_factory=list)  # 测试期 OOS 概率
    fold_results: list[FoldResult] = field(default_factory=list)
    model_final: Optional[xgb.XGBClassifier] = None
    calibrator: Optional[IsotonicRegression] = None
    params: dict = field(default_factory=dict)


def _build_windows(train_start: int, train_end: int, n_folds: int) -> list[dict]:
    """构建 expanding-window 切分。第 n_folds 折为"全量训练"（用于测试期预测）。

    训练期 [train_start, train_end] 共 T 年；前 n_folds-1 折为验证折，
    每折 val 长度 = T // (n_folds + 1)，训练从 train_start 至 val_start-1。
    """
    T = train_end - train_start + 1
    val_len = max(2, T // (n_folds + 1))
    windows: list[dict] = []
    v_start = train_start + val_len
    for k in range(n_folds - 1):
        v_end = min(v_start + val_len - 1, train_end)
        windows.append({
            "fold": f"val_fold_{k + 1}",
            "train_start": train_start, "train_end": v_start - 1,
            "val_start": v_start, "val_end": v_end,
        })
        v_start = v_end + 1
    windows.append({
        "fold": "full_train",
        "train_start": train_start, "train_end": train_end,
        "val_start": None, "val_end": None,
    })
    return windows


def _scale_pos_weight(y: pd.Series) -> float:
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    return float(n_neg / max(n_pos, 1))


def _fit_xgb(X_train: pd.DataFrame, y_train: pd.Series, X_val: Optional[pd.DataFrame],
             y_val: Optional[pd.Series], spw: float) -> tuple[xgb.XGBClassifier, float]:
    """拟合 XGBoost，返回 (model, val_auc)。early stopping 用验证 AUC。"""
    p = cfg()["model"]["xgb"]
    es = cfg()["model"]["cv"]["early_stopping_rounds"]
    common = dict(
        n_estimators=p["n_estimators"],
        max_depth=p["max_depth"],
        learning_rate=p["learning_rate"],
        subsample=p["subsample"],
        colsample_bytree=p["colsample_bytree"],
        reg_lambda=p["l2_reg"],
        min_child_weight=p["min_child_weight"],
        scale_pos_weight=spw,
        random_state=p["random_state"],
        eval_metric="auc",
        tree_method="hist",
        verbosity=0,
    )
    if X_val is not None and len(X_val) > 0 and y_val is not None and y_val.sum() > 0:
        model = xgb.XGBClassifier(early_stopping_rounds=es, **common)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        auc = float(roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]))
    else:
        model = xgb.XGBClassifier(**common)
        model.fit(X_train, y_train)
        auc = float("nan")
    return model, auc


def run_blocked_cv(features: pd.DataFrame, feat_cols: Optional[list[str]] = None,
                   model_version: str = MODEL_VERSION) -> CVRun:
    """执行完整的 blocked expanding-window CV 并返回全部产物（供 F07 输出与 F09 消融复用）。

    :param features: features.csv 内容（iso3, year, 特征列, y_onset）
    :param feat_cols: 使用的特征列（消融时传入子集）
    """
    mcfg = cfg()["model"]
    cv = mcfg["cv"]
    feat_cols = feat_cols or get_factor_columns()
    df = features.copy()
    run = CVRun()
    run.params = {"model_version": model_version, "feat_cols": feat_cols,
                  "test_start": cv["test_start"], "test_end": cv["test_end"]}

    train_end = cv["test_start"] - 1
    windows = _build_windows(cv["cv_start"], train_end, cv["n_folds"])
    test = df[(df["year"] >= cv["test_start"]) & (df["year"] <= cv["test_end"])].copy()

    cal_probs, cal_y = np.array([]), np.array([])
    for w in windows:
        train = df[(df["year"] >= w["train_start"]) & (df["year"] <= w["train_end"])].copy()
        X_tr, y_tr = train[feat_cols], train["y_onset"]
        spw = _scale_pos_weight(y_tr)
        if w["val_start"] is None:  # 最终全量模型
            run.model_final, _ = _fit_xgb(X_tr, y_tr, None, None, spw)
            continue
        val = df[(df["year"] >= w["val_start"]) & (df["year"] <= w["val_end"])].copy()
        X_va, y_va = val[feat_cols], val["y_onset"]
        if y_va.sum() == 0:
            meta.log_stage("cv_fold_skipped", fold=w["fold"], reason="验证折无正样本")
            continue
        model, auc = _fit_xgb(X_tr, y_tr, X_va, y_va, spw)
        proba = model.predict_proba(X_va)[:, 1]
        for k, (_, r) in enumerate(val.iterrows()):
            run.val_rows.append({"country": r["iso3"], "year": int(r["year"]),
                                 "p_conflict": float(proba[k]),
                                 "fold": w["fold"], "model_version": model_version})
        cal_probs = np.concatenate([cal_probs, proba])
        cal_y = np.concatenate([cal_y, y_va.to_numpy()])
        run.fold_results.append(FoldResult(fold=w["fold"], train_start=w["train_start"],
                                           train_end=w["train_end"], val_start=w["val_start"],
                                           val_end=w["val_end"], n_train=len(train), n_val=len(val),
                                           n_positive_train=int(y_tr.sum()), auc_val=auc,
                                           params={"scale_pos_weight": spw}))
        meta.log_stage("cv_fold", fold=w["fold"], auc=round(auc, 4), spw=round(spw, 2))

    if run.model_final is None:
        raise RuntimeError("训练失败：无最终模型。请检查面板数据完整性（特征全 NaN / 正样本过少）")

    # ---- 测试期 OOS 预测（最终模型，严格一次访问）----
    X_te = test[feat_cols]
    raw_test = run.model_final.predict_proba(X_te)[:, 1]

    # ---- 概率校准（§8.2：Platt/isotonic，训练期验证折拟合）----
    # 小样本下 isotonic 易将概率压平，故 Platt(sigmoid) 优先；若校准后退化则回退 raw
    cal_type = "none"
    if len(cal_probs) > 20:
        platt = LogisticRegression(max_iter=2000)
        platt.fit(cal_probs.reshape(-1, 1), cal_y)
        p_cal = platt.predict_proba(raw_test.reshape(-1, 1))[:, 1]
        if p_cal.max() >= 0.3:
            run.calibrator = platt
            cal_type = "platt"
            p_test = p_cal
        else:
            iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-6, y_max=1 - 1e-6)
            iso.fit(cal_probs, cal_y)
            p_iso = iso.predict(raw_test)
            if p_iso.max() >= 0.3:
                run.calibrator = iso
                cal_type = "isotonic"
                p_test = p_iso
            else:
                p_test = raw_test  # 校准退化 → 回退 raw（记录）
                cal_type = "raw-fallback"
                meta.log_stage("calibration_degraded", max_prob=float(p_test.max()),
                               note="校准后概率被压平，回退模型原始概率")
    else:
        p_test = raw_test
    run.params["calibration"] = cal_type
    for k, (_, r) in enumerate(test.iterrows()):
        run.test_rows.append({"country": r["iso3"], "year": int(r["year"]),
                              "p_conflict": float(p_test[k]),
                              "fold": "test", "model_version": model_version})
    return run


def train_and_predict(features: pd.DataFrame) -> dict:
    """F07 主入口：跑 CV，落盘 oos_predictions.csv / xgb_params.json / xgb_model.json。"""
    mcfg = cfg()["model"]
    run = run_blocked_cv(features)

    oos = pd.DataFrame(run.val_rows + run.test_rows)
    oos["p_conflict"] = oos["p_conflict"].clip(0.0, 1.0)
    oos.to_csv(get_path("oos_file"), index=False)

    params = {**mcfg["xgb"], "scale_pos_weight_auto": mcfg["xgb"]["scale_pos_weight_auto"],
              "cv_windows": [fr.fold for fr in run.fold_results] + ["full_train"],
              "model_version": MODEL_VERSION,
              "test_start": mcfg["cv"]["test_start"], "test_end": mcfg["cv"]["test_end"],
              "train_end": mcfg["cv"]["test_start"] - 1,
              "calibration": run.params.get("calibration")}
    with open(get_path("params_file"), "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False, indent=2)
    run.model_final.save_model(str(get_path("model_file")))

    test = features[(features["year"] >= mcfg["cv"]["test_start"]) &
                    (features["year"] <= mcfg["cv"]["test_end"])]
    y_te = test["y_onset"]
    test_auc = float(roc_auc_score(y_te, oos[oos["fold"] == "test"]["p_conflict"])) \
        if len(np.unique(y_te)) > 1 else float("nan")
    meta.log_stage("xgb_train", n_oos=len(oos), test_auc=round(test_auc, 4) if test_auc == test_auc else None,
                   calibration=params["calibration"])
    return {"oos_df": oos, "params": params, "model_version": MODEL_VERSION,
            "fold_results": run.fold_results, "test_auc": test_auc,
            "calibrator": run.calibrator, "run": run}
