"""tests/test_pipeline.py：管道级测试（PRD §10.2 / §13.4）。

不依赖外部网络（使用内置模拟数据），验证端到端可运行与关键产物。
"""
import json

import pytest

from conflict_ew import meta
from conflict_ew.paths import get_path


@pytest.fixture(scope="module")
def pipeline_result():
    """跑通全管道（验证编排逻辑与产物完整性）。"""
    from conflict_ew.pipeline import run_all

    summary = run_all(force_refresh=False, generate_report_after=False, export_web=False)
    return summary


def test_pipeline_produces_core_artifacts(pipeline_result):
    assert pipeline_result["panel"]["n_rows"] > 0
    assert pipeline_result["panel"]["n_countries"] > 50
    assert pipeline_result["panel"]["audit_passed"] is True
    assert 0.005 <= pipeline_result["panel"]["positive_rate"] <= 0.10

    for key in ("panel_file", "kalman_signals_file", "features_file",
                "oos_file", "shap_file", "metrics_file"):
        assert get_path(key).exists(), f"缺少产物: {key}"


def test_oos_predictions_cover_test_period(pipeline_result):
    import pandas as pd

    oos = pd.read_csv(get_path("oos_file"))
    test = oos[oos["fold"] == "test"]
    assert len(test) > 100
    assert test["year"].min() >= 2015
    assert test["year"].max() <= 2024
    assert test["p_conflict"].between(0, 1).all()


def test_metrics_json_structure(pipeline_result):
    with open(get_path("metrics_file"), "r", encoding="utf-8") as f:
        m = json.load(f)
    assert "auc" in m and "ci_lower" in m["auc"]
    assert "brier" in m and "reliability" in m["brier"]
    assert "ablation" in m and len(m["ablation"]) == 4
    assert "baseline_comparison" in m


def test_manifest_has_sha256(pipeline_result):
    mf = meta.load_manifest()
    entries = mf.get("entries", {})
    assert any(v.get("sha256") for v in entries.values())
