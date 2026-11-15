"""Golden 值回归：论文数值断言（真实数据栈 A2 基准）。

数据栈（R-11-A2，2026-11-15 固化）：
  军费/宏观 = WDI 真实抓取；标签 = UCDP intrastate onset v25.1（真实）；
  Polity = Polity5 官方原件；邻接 = COW Direct Contiguity v3.2。
  （ICOW/ATOP/V-Dem 官方源不可脚本直连 → F5/F6/vdem_ldi 降级为 0，见 PROVENANCE.md）

断言值 = 本数据栈 run_all.py 的可复现产物（artifacts/metrics.json）。
旧模拟栈数值（AUC 0.5379/0.5497/0.5909 等）已随 A2 决策废止——如需复现旧栈，
设 config.data.external_source: simulated（数值见 git 历史 1ca26f0 的测试文件）。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conflict_ew.baselines import baseline_logistic_re, baseline_var, paired_bootstrap
from conflict_ew.factors import get_factor_columns
from conflict_ew.paths import get_path


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return pd.read_csv(get_path("features_file"))


def test_logistic_re_golden(features: pd.DataFrame) -> None:
    b = baseline_logistic_re(features, get_factor_columns())
    assert b["auc"] == pytest.approx(0.5716, abs=5e-4)
    assert b["auc_ci_delong"][0] == pytest.approx(0.4897, abs=5e-4)
    assert b["auc_ci_delong"][1] == pytest.approx(0.6534, abs=5e-4)
    assert b["tau"] == pytest.approx(0.5369, abs=5e-4)
    assert b["tau_quantile"] == pytest.approx(0.695, abs=2e-3)
    assert b["flagged"] == 448
    assert b["recall_at_tau"] == pytest.approx(0.4590, abs=5e-4)
    assert b["precision_at_tau"] == pytest.approx(0.0625, abs=5e-4)
    assert b["brier_raw"] == pytest.approx(0.2647, abs=5e-4)
    assert b["partial_auc_0_0.1"] == pytest.approx(0.0548, abs=5e-4)
    assert b["lead_time_months"] == pytest.approx(33.0, abs=0.05)
    assert b["onsets_covered"] == "32/61"
    assert b["n"] == 1650


def test_var2_golden(features: pd.DataFrame) -> None:
    b = baseline_var(features)
    assert b["auc"] == pytest.approx(0.4479, abs=5e-4)
    assert b["auc_ci_delong"][0] == pytest.approx(0.3739, abs=5e-4)
    assert b["auc_ci_delong"][1] == pytest.approx(0.5219, abs=5e-4)
    assert b["tau"] == pytest.approx(0.5156, abs=5e-4)
    assert b["tau_quantile"] == pytest.approx(0.345, abs=2e-3)
    assert b["flagged"] == 1614
    assert b["recall_at_tau"] == pytest.approx(1.0, abs=5e-4)
    assert b["precision_at_tau"] == pytest.approx(0.0378, abs=5e-4)
    assert b["brier_raw"] == pytest.approx(0.2649, abs=5e-4)
    assert b["partial_auc_0_0.1"] == pytest.approx(0.0080, abs=5e-4)
    assert b["lead_time_months"] == pytest.approx(39.9, abs=0.05)
    assert b["onsets_covered"] == "61/61"
    assert b["n"] == 1633


def test_paired_diff_golden(features: pd.DataFrame) -> None:
    """配对差异（真实数据栈：KF+MF − LogReg_RE = −0.027 p=0.614；
    KF+MF − VAR2 = +0.098 p=0.018）。依赖 artifacts/oos_predictions.csv。"""
    oos_path = get_path("oos_file")
    if not oos_path.exists():
        pytest.skip("oos_predictions.csv 不存在（先跑 run_all.py）")
    oos = pd.read_csv(oos_path)
    test = oos[oos["fold"] == "test"]
    lab = {(r["iso3"], int(r["year"])): int(r["y_onset"])
           for _, r in features.iterrows()}
    keys = list(zip(test["country"], test["year"]))
    y_kf = np.array([lab[k] for k in keys])
    p_kf = test["p_conflict"].to_numpy()
    for b, gold_diff, gold_p in ((baseline_logistic_re(features, get_factor_columns()),
                                  -0.0269, 0.614),
                                 (baseline_var(features), 0.0978, 0.018)):
        pm = {(r["country"], r["year"]): r["p_conflict"] for r in b["test_rows"]}
        mask = np.array([k in pm for k in keys])
        d = paired_bootstrap(y_kf[mask], p_kf[mask],
                             np.array([pm[k] for k, m in zip(keys, mask) if m]))
        assert d["diff"] == pytest.approx(gold_diff, abs=1e-3)
        assert d["p_value"] == pytest.approx(gold_p, abs=5e-3)
