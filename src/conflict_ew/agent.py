"""Agent 推理链生成（规则模板 Mock）。

PRD §8.3：LLM 后端为 Phase2 可选（rule-template | llama3 | gpt4，config.yaml 切换）。
§13.4：LLM 后端未接入前用规则模板 Mock 推理链（延迟 ≤1s）。

三机制 grounding（R4 应对）：
  1. 数值预计算：所有引用数字来自特征/SHAP/概率，不自由发挥
  2. 因子级引用：每步推理标注 F1-F6 因子 ID 与观测期
  3. 核查清单：高风险输出附人工复核清单

伦理约束（§9/§11）：输出概率 + 显式不确定性；不输出军事行动建议；自证预言风险声明。
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

from . import meta
from .paths import cfg, get_path

FACTOR_NAMES = {
    "F1": "军费异常信号 e*",
    "F2": "军备竞赛",
    "F3": "经济压力",
    "F4": "政治不稳",
    "F5": "领土争端",
    "F6": "同盟关系",
}


def risk_level_of(p: float, thresholds: Optional[dict] = None) -> str:
    """三级风险分级（0.7/0.5 默认阈值）。"""
    t = thresholds or cfg()["eval"]["thresholds"]
    if p >= t["high"]:
        return "high"
    if p >= t["mid"]:
        return "mid"
    return "low"


RISK_LABEL = {"high": "高风险", "mid": "中风险", "low": "低风险"}


def _direction_statement(factor_id: str, value: float, shap: float) -> str:
    """单因子规则模板：生成自然语言说明（引用数值 + 观测期）。"""
    v = value
    if factor_id == "F1":
        if abs(v) >= 2:
            return f"军费支出显著偏离历史趋势（|e*|={abs(v):.2f}≥2），军事意图信号异常，对风险概率的贡献为 {shap:+.3f}"
        return f"军费支出基本符合历史趋势（e*={v:+.2f}），未出现异常抬升信号"
    if factor_id == "F2":
        if v > 0.02:
            return f"邻国军费增速超过本国（增速差={v:+.3f}），区域军备竞赛压力上升"
        return f"邻国军费增速与本国接近或更低（增速差={v:+.3f}），军备竞赛压力有限"
    if factor_id == "F3":
        parts = []
        if v is not None and abs(v) > 0.5:
            parts.append(f"经济压力因子组合贡献 {shap:+.3f}")
        return "；".join(parts) if parts else "经济压力因子贡献不显著（接近零）"
    if factor_id == "F4":
        if v is not None and abs(v) > 0.3:
            return f"政治制度指标出现明显波动（变化量={v:+.2f}），政治不稳定信号增强"
        return "政治制度指标保持稳定，政治不稳信号不明显"
    if factor_id == "F5":
        return "存在活跃领土主张（ICOW 记录），领土争端构成潜在冲突诱因" if v >= 0.5 \
            else "无活跃领土主张（ICOW 记录），领土争端因素不显著"
    if factor_id == "F6":
        return "正式同盟义务发生变化（ATOP 记录），区域权力平衡正在调整" if v >= 0.5 \
            else "同盟义务未发生明显变化，同盟因素稳定"
    return f"因子 {factor_id} 观测值 {v:+.3f}，贡献 {shap:+.3f}"


def generate_reasoning_chain(country: str, year: int, p_conflict: float,
                             ci: Optional[dict] = None,
                             factors: Optional[dict[str, float]] = None,
                             shap_values: Optional[dict[str, float]] = None,
                             e_star: Optional[float] = None,
                             thresholds: Optional[dict] = None) -> dict:
    """F15/§3 推理链生成：返回结构化推理 JSON。

    :param country: iso3
    :param year: 观测年份
    :param p_conflict: 冲突风险概率 [0,1]
    :param ci: 95% CI（可选）
    :param factors: F1-F6 因子值
    :param shap_values: F1-F6 SHAP 贡献
    :param e_star: 军费异常信号（冗余传参，兼容旧调用）
    """
    t0 = time.time()
    factors = factors or {}
    shap_values = shap_values or {}
    level = risk_level_of(p_conflict, thresholds)
    steps = []
    for fid in ("F1", "F2", "F3", "F4", "F5", "F6"):
        if fid not in factors:
            continue
        val = factors.get(fid)
        sh = shap_values.get(fid, 0.0)
        statement = _direction_statement(fid, val, sh)
        steps.append({
            "factor_id": fid,
            "factor_name": FACTOR_NAMES[fid],
            "value": round(float(val), 4) if val is not None else None,
            "shap_contribution": round(float(sh), 4),
            "is_substantial": abs(sh) >= cfg()["model"]["shap"]["substantial_threshold"],
            "observation_period": f"截至 {year} 年",
            "statement": statement,
        })
    # 排序：贡献绝对值降序
    steps.sort(key=lambda s: abs(s["shap_contribution"]), reverse=True)

    top = [s for s in steps if s["is_substantial"]][:3]
    top_txt = "；".join(f"{s['factor_name']}({s['shap_contribution']:+.3f})" for s in top) or "无实质驱动因子"
    level_txt = RISK_LABEL[level]
    summary = (
        f"{country} {year} 年冲突风险概率 {p_conflict:.2f}（95% CI: "
        f"{ci['ci_lower']:.2f}-{ci['ci_upper']:.2f}" if ci else f"{country} {year} 年冲突风险概率 {p_conflict:.2f}"
    )
    summary += f"），判定为{level_txt}。主要驱动因子：{top_txt}。"
    if p_conflict >= (thresholds or cfg()["eval"]["thresholds"])["high"]:
        summary += " 建议进入人工复核流程，核查清单已生成。"
    summary += " 本判断基于统计模型，存在不确定性，不构成军事行动建议。"

    chain = {
        "country": country,
        "year": year,
        "probability": round(float(p_conflict), 4),
        "ci": ci,
        "risk_level": level,
        "risk_level_label": level_txt,
        "steps": steps,
        "summary": summary,
        "checklist": cfg()["alert"]["agent_checklist"] if level == "high" else [],
        "backend": cfg()["llm"]["backend"],
        "disclaimer": "概率化输出+显式不确定性：本系统不输出政治判断或军事行动建议，部署判断权归人（对齐论文 §2.5 伦理约束）。",
        "latency_ms": round((time.time() - t0) * 1000, 1),
    }
    return chain


def generate_for_oos(oos_test: pd.DataFrame, shap_df: pd.DataFrame,
                     factors: pd.DataFrame, ci_level: float = 0.95) -> list[dict]:
    """对测试期全部 (country, year) 生成推理链（供 P04 国家详情 / 报告 / 预警）。"""
    chains = []
    shap_pivot = shap_df.pivot_table(index=["country", "year"], columns="factor_id",
                                     values="shap_value").reset_index()
    fac_pivot = factors.set_index(["iso3", "year"])
    from scipy.stats import norm

    z = norm.ppf(1 - (1 - ci_level) / 2)
    for _, r in oos_test.iterrows():
        p = float(r["p_conflict"])
        # 简化 CI：经验近似（logit 空间 ±z·0.25，再映射回概率）
        logit = np_logit(p)
        lo = np_expit(logit - z * 0.25)
        hi = np_expit(logit + z * 0.25)
        ci = {"ci_lower": float(lo), "ci_upper": float(hi), "level": ci_level,
              "method": "logit 近似（经验 SE≈0.25）"}
        shp = {}
        row = shap_pivot[(shap_pivot["country"] == r["country"]) & (shap_pivot["year"] == int(r["year"]))]
        if len(row):
            for f in ("F1", "F2", "F3", "F4", "F5", "F6"):
                if f in row.columns and row.iloc[0][f] == row.iloc[0][f]:
                    shp[f] = float(row.iloc[0][f])
        facs = {}
        try:
            fr = fac_pivot.loc[(r["country"], int(r["year"]))]
            for c in factors.columns:
                if c.startswith("F") and "_" in c:
                    key = c.split("_")[0]
                    if fr[c] == fr[c]:
                        facs.setdefault(key, 0.0)
                        if c.split("_")[0] in ("F3", "F4"):
                            facs[key] += float(fr[c])
                        else:
                            facs[key] = float(fr[c])
        except KeyError:
            pass
        chains.append(generate_reasoning_chain(r["country"], int(r["year"]), p, ci, facs, shp))
    return chains


def np_logit(p: float) -> float:
    import numpy as np

    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(np.log(p / (1 - p)))


def np_expit(x: float) -> float:
    import numpy as np

    return float(1 / (1 + np.exp(-x)))


def save_chains(chains: list[dict], path=None) -> None:
    path = path or (get_path("artifact_dir") / "reasoning_chains.json")
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"n": len(chains), "backend": cfg()["llm"]["backend"], "chains": chains},
                  f, ensure_ascii=False, indent=1)
    meta.log_stage("agent_chains", n=len(chains), backend=cfg()["llm"]["backend"])
