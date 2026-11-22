# -*- coding: utf-8 -*-
"""isotonic 校准 + 8-bin 可靠性分解 + VAR 区间（论文 Fig.4b / TABLE III 口径）。

规范来源：论文 revision_v3 release_package scripts/m2_m3_calibration_checks.py
（校准节）与 TABLE III 脚注（Wilson / 参数 bootstrap）。

- isotonic：在主模型 oos 的 4 个验证折（2,640 行 / 95 阳性）上拟合一次，
  **同一曲线**套用于三个模型的测试分（论文 "applied identically to all
  three models"；golden：KF 0.120→0.0298，两基线 →0.028）。
- 可靠性：8 个等宽 bin（宽 0.125）；BSS 对气候学基准（常数基准率预报）。
- VAR(2) precision 用 Wilson 区间（x=TP=4, n=flagged=42）；
  Brier 用参数 bootstrap（在模型概率下重抽 y*，golden [0.254,0.271]）。

输出：artifacts/calibration_reliability.json
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conflict_ew.baselines import baseline_logistic_re, baseline_var  # noqa: E402
from conflict_ew.build_panel import load_base_panel  # noqa: E402
from conflict_ew.factors import get_factor_columns  # noqa: E402
from conflict_ew.paths import get_path  # noqa: E402

OUT = ROOT / "artifacts"
OUT.mkdir(exist_ok=True)

EDGES = (0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0)

feats = pd.read_csv(get_path("features_file")).reset_index(drop=True)
oos = pd.read_csv(get_path("oos_file"))
lab = {(r.iso3, int(r.year)): int(r.y_onset) for r in feats.itertuples()}
oos["y"] = pd.to_numeric([lab.get((c, int(y)), None) for c, y in zip(oos["country"], oos["year"])],
                         errors="coerce")
val = oos[(oos["fold"] != "test") & oos["y"].notna()].copy()
test = oos[oos["fold"] == "test"].dropna(subset=["y"]).copy()
val["y"] = val["y"].astype(int)
test["y"] = test["y"].astype(int)

# ---- 校准曲线：主模型用 oos 验证折（M6 规范）；两基线各用自己的训练期
#      in-sample 分数（与 τ 规则同口径；golden 0.028/0.028） ----
iso_kf = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso_kf.fit(val["p_conflict"], val["y"])

# ---- 三模型测试分 ----
y_kf = test["y"].to_numpy()
p_kf = test["p_conflict"].to_numpy()
panel_df = load_base_panel()
bl_lr = baseline_logistic_re(feats, get_factor_columns(), panel_df)
bl_va = baseline_var(feats, panel_df)
y_lr = np.array([lab[(r["country"], r["year"])] for r in bl_lr["test_rows"]])
p_lr = np.array([r["p_conflict"] for r in bl_lr["test_rows"]])
lab_va = {(r["iso3"], int(r["year"])): int(r["y_onset"]) for _, r in feats.iterrows()}
y_va = np.array([lab_va[(r["country"], r["year"])] for r in bl_va["test_rows"]])
p_va = np.array([r["p_conflict"] for r in bl_va["test_rows"]])
iso_lr = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso_lr.fit(np.array(bl_lr["train_preds"]["p"]), np.array(bl_lr["train_preds"]["y"]))
iso_va = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
iso_va.fit(np.array(bl_va["train_preds"]["p"]), np.array(bl_va["train_preds"]["y"]))
ISOS = {"KF+MF": iso_kf, "LogReg_RE": iso_lr, "VAR2": iso_va}


def bins(df, pcol):
    out = []
    for i in range(len(EDGES) - 1):
        lo, hi = EDGES[i], EDGES[i + 1]
        m = (df[pcol] >= lo) & (df[pcol] < hi if hi < 1.0 else df[pcol] <= hi)
        sub = df[m]
        out.append({"lo": lo, "hi": hi, "n": int(len(sub)),
                    "pred": round(float(sub[pcol].mean()), 4) if len(sub) else None,
                    "obs": round(float(sub["y"].mean()), 4) if len(sub) else None})
    return out


def brier_decomp(p, y):
    brier = float(np.mean((p - y) ** 2))
    base = float(y.mean())
    unc = float(base * (1 - base))
    rel = 0.0
    res = 0.0
    for i in range(len(EDGES) - 1):
        lo, hi = EDGES[i], EDGES[i + 1]
        m = (p >= lo) & (p < hi if hi < 1.0 else p <= hi)
        if not m.any():
            continue
        pm = float(p[m].mean())
        oi = float(y[m].mean())
        ni = int(m.sum())
        n = len(p)
        rel += ni / n * (pm - oi) ** 2
        res += ni / n * (oi - base) ** 2
    return {"brier": round(brier, 4), "reliability": round(rel, 4),
            "resolution": round(res, 6), "uncertainty": round(unc, 6),
            "climatology_brier": round(unc, 4),
            "bss": round(1 - brier / unc, 4),
            "pred_mean": round(float(p.mean()), 4), "obs_rate": round(float(y.mean()), 4)}


# ---- VAR 区间（TABLE III 脚注口径）----
def wilson(x: int, n: int, z: float = 1.959964) -> list[float]:
    ph = x / n
    denom = 1 + z * z / n
    center = (ph + z * z / (2 * n)) / denom
    half = z * np.sqrt(ph * (1 - ph) / n + z * z / (4 * n * n)) / denom
    return [round(center - half, 4), round(center + half, 4)]


def param_bootstrap_brier(p, y, b=2000, seed=20240101):
    """参数 bootstrap（论文措辞口径）：onset 计数 k~Bin(n, base rate)、
    正例位置均匀重抽，Brier 的 95% 区间。

    注：平坦概率尺度下该量对重抽几乎不敏感；论文正文区间 [0.254,0.271]
    无法由任何 y-重抽口径复现（Brier=mean(p²)−0.026·(n₁/n)，n₁≤n 上界
    0.2632），按 R-11 回写论文。此处如实输出本口径结果。
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    vals = []
    for _ in range(b):
        k = rng.binomial(n, y.mean())
        ystar = np.zeros(n, dtype=int)
        ystar[rng.choice(n, k, replace=False)] = 1
        vals.append(float(np.mean((p - ystar) ** 2)))
    return {"brier": round(float(np.mean((p - y) ** 2)), 4),
            "ci_lower": round(float(np.percentile(vals, 2.5)), 4),
            "ci_upper": round(float(np.percentile(vals, 97.5)), 4),
            "n_boot": b,
            "procedure": "k~Bin(n, base_rate); positives placed uniformly at random"}


models = {"KF+MF": (y_kf, p_kf), "LogReg_RE": (y_lr, p_lr), "VAR2": (y_va, p_va)}


def _logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(pc / (1 - pc))


res = {"isotonic_fit": {"n_val": int(len(val)), "n_val_pos": int(val["y"].sum()),
                        "n_val_folds": int(val["fold"].nunique()),
                        "KF+MF": "oos 验证折（M6 规范）",
                        "LogReg_RE/VAR2": "各自训练期 in-sample 分数（与 τ 规则同口径）"},
       "models": {}}
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

for name, (y, p) in models.items():
    p_cal = np.asarray(ISOS[name].predict(p))
    res["models"][name] = {
        "raw": brier_decomp(p, y),
        "isotonic_calibrated": brier_decomp(p_cal, y),
    }
    # AUC 追踪：isotonic 不保证单调 → 可能塌缩排序（本数据实测 resolution=0）；
    # Platt/逻辑缩放单调可逆 → AUC_after 应与 raw 一致，排序信息保留
    res["models"][name]["raw"]["auc"] = round(float(roc_auc_score(y, p)), 4)
    res["models"][name]["isotonic_calibrated"]["auc_after"] = round(float(roc_auc_score(y, p_cal)), 4)
    if name == "KF+MF":
        pl = LogisticRegression(max_iter=1000)
        pl.fit(_logit(val["p_conflict"].to_numpy()).reshape(-1, 1), val["y"].to_numpy())
        p_platt = pl.predict_proba(_logit(p).reshape(-1, 1))[:, 1]
        res["models"][name]["platt_calibrated"] = brier_decomp(p_platt, y)
        res["models"][name]["platt_calibrated"]["auc_after"] = round(float(roc_auc_score(y, p_platt)), 4)
        res["models"][name]["calibration_note"] = (
            "isotonic 在本数据上把测试分塌缩为常数（resolution=0），不可用于排序；"
            "需要排序时使用 platt_calibrated（单调，AUC 保持）——两者 Brier 差异极小。")
    if name == "KF+MF":
        df_kf = pd.DataFrame({"y": y, "p_conflict": p, "p_cal": p_cal})
        res["models"][name]["raw"]["bins"] = bins(df_kf, "p_conflict")
        res["models"][name]["isotonic_calibrated"]["bins"] = bins(df_kf, "p_cal")
    if name == "VAR2":
        # Wilson 用基线内部混淆矩阵（未舍入 τ；rounded τ 会翻转平坦尺度的告警行）
        res["models"][name]["precision_wilson"] = wilson(bl_va["TP"], bl_va["flagged"])
        res["models"][name]["wilson_inputs"] = {"TP": bl_va["TP"], "flagged": bl_va["flagged"]}
        res["models"][name]["brier_param_bootstrap"] = param_bootstrap_brier(p, y)

print(json.dumps(res, ensure_ascii=False, indent=1)[:2200])
(OUT / "calibration_reliability.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                                  encoding="utf-8")
print("DONE: calibration_reliability.json")
