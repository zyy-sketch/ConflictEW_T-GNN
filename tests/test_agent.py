"""tests/test_agent.py：Agent 推理链（PRD §13.4 新增 / R4 应对）。"""
import json

from conflict_ew.agent import generate_reasoning_chain, np_expit, np_logit, risk_level_of


def test_risk_level_thresholds():
    assert risk_level_of(0.9, {"high": 0.7, "mid": 0.5}) == "high"
    assert risk_level_of(0.7, {"high": 0.7, "mid": 0.5}) == "high"  # 临界 0.7
    assert risk_level_of(0.6, {"high": 0.7, "mid": 0.5}) == "mid"
    assert risk_level_of(0.5, {"high": 0.7, "mid": 0.5}) == "mid"  # 临界 0.5
    assert risk_level_of(0.3, {"high": 0.7, "mid": 0.5}) == "low"


def test_chain_structure_and_grounding():
    chain = generate_reasoning_chain(
        "XXX", 2024, 0.82,
        ci={"ci_lower": 0.6, "ci_upper": 0.92},
        factors={"F1": 2.6, "F2": 0.05, "F3": -1.2, "F4": -0.4, "F5": 1.0, "F6": 0.0},
        shap_values={"F1": 0.9, "F2": 0.2, "F3": -0.3, "F4": -0.1, "F5": 0.4, "F6": 0.0},
    )
    assert chain["risk_level"] == "high"
    assert len(chain["steps"]) == 6
    # 三机制 grounding：数值引用 + 因子级引用 + 核查清单
    assert "2.60" in chain["summary"] or "0.9" in chain["summary"]
    assert chain["steps"][0]["observation_period"] == "截至 2024 年"
    assert len(chain["checklist"]) > 0  # 高风险触发核查清单
    # 伦理声明：不输出军事行动建议
    assert "军事行动建议" in chain["disclaimer"]
    assert chain["backend"] == "rule-template"
    assert chain["latency_ms"] <= 1000  # §13.4 Mock 延迟 ≤1s


def test_chain_low_risk_no_checklist():
    chain = generate_reasoning_chain(
        "YYY", 2024, 0.1,
        factors={"F1": 0.1, "F2": 0.0, "F3": 0.5, "F4": 0.2, "F5": 0.0, "F6": 0.0},
        shap_values={},
    )
    assert chain["risk_level"] == "low"
    assert chain["checklist"] == []


def test_logit_roundtrip():
    assert abs(np_expit(np_logit(0.3)) - 0.3) < 1e-9
    assert abs(np_expit(np_logit(0.9)) - 0.9) < 1e-9


def test_chain_json_serializable():
    chain = generate_reasoning_chain("ZZZ", 2023, 0.55, factors={"F1": 0.3}, shap_values={"F1": 0.1})
    json.dumps(chain, ensure_ascii=False)  # 不应抛异常
