"""Golden 值回归：批次③（校准 / 二级指标 / 敏感性）产物断言。

真实数据栈 A2 基准（UCDP 标签 + WDI 军费 + Polity5 + COW 邻接）。
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
    assert d["pooled_pearson_r_all_years"] == pytest.approx(0.926, abs=5e-4)
    assert d["anomaly_flag_agreement_rate"] == pytest.approx(0.9886, abs=5e-4)
    assert d["per_country_pearson_r_median"] == pytest.approx(0.9995, abs=5e-4)
    m3 = _load("m3_price_diagnostic.json")
    assert m3["pooled_corr_e_star_inflation"] == pytest.approx(0.0456, abs=5e-4)
    assert m3["within_country_corr_e_star_inflation"] == pytest.approx(0.0447, abs=5e-4)


def test_calibration_golden() -> None:
    d = _load("calibration_reliability.json")["models"]
    kf = d["KF+MF"]
    assert kf["raw"]["brier"] == pytest.approx(0.0391, abs=5e-4)
    assert kf["raw"]["bss"] == pytest.approx(-0.0985, abs=5e-4)
    assert kf["isotonic_calibrated"]["brier"] == pytest.approx(0.0370, abs=5e-4)
    assert kf["isotonic_calibrated"]["reliability"] == pytest.approx(0.0014, abs=5e-4)
    assert kf["isotonic_calibrated"]["bss"] == pytest.approx(-0.0384, abs=5e-4)
    # 基线校准（各自训练期 in-sample isotonic）
    assert d["LogReg_RE"]["isotonic_calibrated"]["brier"] == pytest.approx(0.0355, abs=5e-4)
    assert d["VAR2"]["isotonic_calibrated"]["brier"] == pytest.approx(0.0361, abs=5e-4)
    # Wilson（VAR2 真实标签：TP=61, flagged=1614）
    lo, hi = d["VAR2"]["precision_wilson"]
    assert lo == pytest.approx(0.0295, abs=5e-4)
    assert hi == pytest.approx(0.0482, abs=5e-4)
    # 参数 bootstrap：平坦尺度下区间极窄（真实标签 [0.2646,0.2652]）
    pb = d["VAR2"]["brier_param_bootstrap"]
    assert pb["ci_lower"] <= pb["brier"] <= pb["ci_upper"]


def test_secondary_metrics_golden() -> None:
    d = _load("n4_bootstrap_ci.json")
    gold = {"KF+MF": (0.0311, 0.0087, 0.0984, 0.06, 0.02, 0.11),
            "LogReg_RE": (0.0548, 0.0319, 0.1404, 0.04, 0.01, 0.10),
            "VAR2": (0.0080, 0.0000, 0.0842, 0.03, 0.00, 0.07)}
    for m, g in gold.items():
        pa, pk = d[m]["partial_auc_0_0.1"], d[m]["precision_at_100"]
        got = (round(pa["point"], 4), pa["ci_lower"], pa["ci_upper"], pk["point"],
               pk["ci_lower"], pk["ci_upper"])
        assert got == g, f"{m}: {got} != {g}"
    n6 = _load("n6_f3_sensitivity.json")
    assert n6["n_imputed_rows"] == 90
    assert n6["positives_in_imputed"] == 1
    assert n6["KF_MF"]["auc_drop_imputed"] == pytest.approx(0.5507, abs=5e-4)
    assert n6["LogReg_RE"]["auc_drop_imputed"] == pytest.approx(0.5713, abs=5e-4)
