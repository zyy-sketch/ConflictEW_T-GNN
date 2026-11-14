"""Golden 值回归：论文 Table III 基线数值断言。

断言值 = 论文 `Multi_revised_v4_submission.docx` TABLE III / §V 正文数值
（来源：revision_v3 release_package artifacts/table3_unified.json）。
任何重跑偏离都应触发人工核查——禁止为凑数调整实现。
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conflict_ew.baselines import (baseline_logistic_re, baseline_var,
                                   paired_bootstrap)
from conflict_ew.factors import get_factor_columns
from conflict_ew.paths import get_path


@pytest.fixture(scope="module")
def features() -> pd.DataFrame:
    return pd.read_csv(get_path("features_file"))


def test_logistic_re_golden(features: pd.DataFrame) -> None:
    b = baseline_logistic_re(features, get_factor_columns())
    assert b["auc"] == pytest.approx(0.5497, abs=5e-4)          # 论文 §V: 0.550
    assert b["auc_ci_delong"][0] == pytest.approx(0.460, abs=5e-4)
    assert b["auc_ci_delong"][1] == pytest.approx(0.6393, abs=5e-4)
    assert b["tau"] == pytest.approx(0.557292, abs=5e-4)        # §V-B: τ=0.557
    assert b["tau_quantile"] == pytest.approx(0.745, abs=2e-3)  # 75th percentile
    assert b["flagged"] == 442                                  # §V-B: 442/1650
    assert b["recall_at_tau"] == pytest.approx(0.3191, abs=5e-4)   # 0.319
    assert b["precision_at_tau"] == pytest.approx(0.0339, abs=5e-4)  # 0.034
    assert b["brier_raw"] == pytest.approx(0.2584, abs=5e-4)    # §V-B: 0.258
    assert b["partial_auc_0_0.1"] == pytest.approx(0.1099, abs=5e-4)  # 0.110
    assert b["lead_time_months"] == pytest.approx(42.8, abs=0.05)
    assert b["onsets_covered"] == "44/47"
    assert b["n"] == 1650


def test_var2_golden(features: pd.DataFrame) -> None:
    b = baseline_var(features)
    assert b["auc"] == pytest.approx(0.5909, abs=5e-4)          # 论文 §V: 0.591
    assert b["auc_ci_delong"][0] == pytest.approx(0.4991, abs=5e-4)
    assert b["auc_ci_delong"][1] == pytest.approx(0.6827, abs=5e-4)
    assert b["tau"] == pytest.approx(0.512981, abs=5e-4)        # §V-B: τ=0.513
    assert b["tau_quantile"] == pytest.approx(0.545, abs=2e-3)  # 55th percentile
    assert b["flagged"] == 42                                   # §V-B: 42 alerts
    assert b["recall_at_tau"] == pytest.approx(0.0851, abs=5e-4)     # TP 4/47
    assert b["precision_at_tau"] == pytest.approx(0.0952, abs=5e-4)  # 0.095
    assert b["brier_raw"] == pytest.approx(0.2624, abs=5e-4)    # §V-B: 0.262
    assert b["partial_auc_0_0.1"] == pytest.approx(0.1117, abs=5e-4)  # 0.112
    assert b["lead_time_months"] == pytest.approx(52.5, abs=0.05)
    assert b["onsets_covered"] == "8/47"
    assert b["n"] == 1633                                       # 1,633 evaluable


def test_paired_diff_golden(features: pd.DataFrame) -> None:
    """配对差异（Table III paired：-0.012 p=0.824 / -0.051 p=0.358）。

    依赖 artifacts/oos_predictions.csv（gitignored）；缺失则跳过。
    """
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
                                  -0.0118, 0.824),
                                 (baseline_var(features), -0.0508, 0.358)):
        pm = {(r["country"], r["year"]): r["p_conflict"] for r in b["test_rows"]}
        mask = np.array([k in pm for k in keys])
        d = paired_bootstrap(y_kf[mask], p_kf[mask],
                             np.array([pm[k] for k, m in zip(keys, mask) if m]))
        assert d["diff"] == pytest.approx(gold_diff, abs=1e-3)
        assert d["p_value"] == pytest.approx(gold_p, abs=5e-3)
