"""Golden 值回归：批次③（校准 / 二级指标 / 敏感性）产物断言。

断言值 = 论文正文与 revision_v3 release_package 权威产物。
产物文件在 artifacts/（gitignored，由 scripts/09–11 生成）；缺失则跳过。
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ART = Path(__file__).resolve().parents[1] / "artifacts"


def _load(name: str) -> dict:
    p = ART / name
    if not p.exists():
        pytest.skip(f"{name} 不存在（先跑 scripts/09–11）")
    return json.loads(p.read_text(encoding="utf-8"))


def test_window_sensitivity_golden() -> None:
    d = _load("m2_window_sensitivity.json")
    assert d["pooled_pearson_r_all_years"] == pytest.approx(0.926, abs=5e-4)   # 论文 §III-A
    assert d["anomaly_flag_agreement_rate"] == pytest.approx(0.9886, abs=5e-4)  # 98.9%
    assert d["per_country_pearson_r_median"] == pytest.approx(0.9995, abs=5e-4)
    m3 = _load("m3_price_diagnostic.json")
    assert m3["pooled_corr_e_star_inflation"] == pytest.approx(0.0456, abs=5e-4)   # 论文 0.046
    assert m3["within_country_corr_e_star_inflation"] == pytest.approx(0.0447, abs=5e-4)  # 0.045


def test_calibration_golden() -> None:
    d = _load("calibration_reliability.json")["models"]
    kf = d["KF+MF"]
    assert kf["raw"]["brier"] == pytest.approx(0.1202, abs=5e-4)          # 论文 0.120
    assert kf["raw"]["bss"] == pytest.approx(-3.3444, abs=5e-4)           # skill -3.344
    assert kf["isotonic_calibrated"]["brier"] == pytest.approx(0.0298, abs=5e-4)  # 0.030
    assert kf["isotonic_calibrated"]["reliability"] == pytest.approx(0.002, abs=5e-4)
    assert kf["isotonic_calibrated"]["bss"] == pytest.approx(-0.0778, abs=5e-4)   # -0.078
    # 基线校准（各自训练期 in-sample isotonic）
    assert d["LogReg_RE"]["isotonic_calibrated"]["brier"] == pytest.approx(0.0277, abs=5e-4)
    assert d["VAR2"]["isotonic_calibrated"]["brier"] == pytest.approx(0.0279, abs=5e-4)
    # 论文正文两者均报 0.028 —— 本口径 0.0277/0.0279，四舍五入一致
    # Wilson（TP=4, flagged=42）
    lo, hi = d["VAR2"]["precision_wilson"]
    assert lo == pytest.approx(0.0377, abs=5e-4)   # 论文 [0.038, 0.221]
    assert hi == pytest.approx(0.2207, abs=5e-4)
    # 参数 bootstrap：论文区间 [0.254,0.271] 数学上不可复现（平坦尺度 Brier
    # = mean(p²)-0.026·(n1/n) ≤ 0.2632），R-11 回写；此处仅断言产物存在且合法
    pb = d["VAR2"]["brier_param_bootstrap"]
    assert pb["ci_lower"] <= pb["brier"] <= pb["ci_upper"]


def test_secondary_metrics_golden() -> None:
    d = _load("n4_bootstrap_ci.json")
    gold = {"KF+MF": (0.0096, 0.0, 0.0724, 0.02, 0.0, 0.04),
            "LogReg_RE": (0.1099, 0.0436, 0.1862, 0.08, 0.03, 0.13),
            "VAR2": (0.1117, 0.0509, 0.2251, 0.07, 0.02, 0.13)}
    for m, g in gold.items():
        pa, pk = d[m]["partial_auc_0_0.1"], d[m]["precision_at_100"]
        got = (pa["point"], pa["ci_lower"], pa["ci_upper"], pk["point"],
               pk["ci_lower"], pk["ci_upper"])
        assert got == g, f"{m}: {got} != {g}"
    n6 = _load("n6_f3_sensitivity.json")
    assert n6["n_imputed_rows"] == 90
    assert n6["positives_in_imputed"] == 4
    assert n6["KF_MF"]["auc_drop_imputed"] == pytest.approx(0.5538, abs=5e-4)     # 论文 0.554
    assert n6["LogReg_RE"]["auc_drop_imputed"] == pytest.approx(0.5329, abs=5e-4)  # 论文 0.533
