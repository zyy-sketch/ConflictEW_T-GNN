"""F15 预警机制：分级预警触发与归档。

- 风险概率穿越阈值触发分级预警：
    p ≥ 0.7        高风险（触发 Agent 核查清单 + 人工复核）
    0.5 ≤ p < 0.7  中风险（持续监控）
    p < 0.5        低风险（常规）
- 概率 NaN 或超 [0,1] 自动拦截 + 记录异常日志（PRD §5.3 阈值越界）
- 通知：站内通知（in-app）；邮件通道预留（需配置 SMTP，默认关闭）
- 阈值来源：config.yaml -> eval.thresholds；阈值选择仅在训练期完成（防泄漏约束）
"""
from __future__ import annotations

import json
import logging
import math
from typing import Optional

import pandas as pd

from . import meta
from .agent import risk_level_of
from .paths import cfg, get_path

logger = logging.getLogger("conflict_ew.alerts")

RISK_LABEL = {"high": "高风险", "mid": "中风险", "low": "低风险"}


def scan_alerts(oos_test: pd.DataFrame, country_names: Optional[dict] = None,
                chains: Optional[list[dict]] = None) -> list[dict]:
    """F15 主入口：扫描测试期 OOS 概率，生成三级预警。

    :param oos_test: fold=='test' 的 OOS 概率（country, year, p_conflict）
    :param country_names: iso3 -> 中文名
    :param chains: 推理链列表（可选，高风险预警附带）
    :return: alerts 列表（按时间/等级排序）
    """
    t = cfg()["eval"]["thresholds"]
    alerts: list[dict] = []
    chain_map = {(c["country"], c["year"]): c for c in (chains or [])}
    seq = 0
    for _, r in oos_test.iterrows():
        p = float(r["p_conflict"])
        # §5.3 阈值越界拦截
        if math.isnan(p) or not (0.0 <= p <= 1.0):
            meta.log_stage("alert_probability_invalid", country=r["country"], year=int(r["year"]),
                           p=p, action="blocked")
            logger.warning("预警拦截：概率 NaN 或超 [0,1] country=%s year=%s", r["country"], r["year"])
            continue
        level = risk_level_of(p, t)
        if level == "low":
            continue
        seq += 1
        chain = chain_map.get((r["country"], int(r["year"])), {})
        alerts.append({
            "alert_id": f"A{20240000 + seq:05d}",
            "country": r["country"],
            "country_name": (country_names or {}).get(r["country"], r["country"]),
            "year": int(r["year"]),
            "p_value": round(p, 4),
            "risk_level": level,
            "risk_level_label": RISK_LABEL[level],
            "trigger_reason": f"风险概率 {p:.2f} 穿越 {t['high'] if level == 'high' else t['mid']} 阈值",
            "trigger_time": meta.utcnow_iso(),
            "status": "pending_review" if level == "high" else "monitoring",
            "agent_checklist": chain.get("checklist", []) if level == "high" else [],
            "summary": chain.get("summary", ""),
            "notification": {"channels": cfg()["alert"]["notify_channels"], "sent": True},
        })
    alerts.sort(key=lambda a: (a["risk_level"] != "high", -a["p_value"]))
    with open(get_path("alerts_file"), "w", encoding="utf-8") as f:
        json.dump({"count": len(alerts), "thresholds": t, "generated_at": meta.utcnow_iso(),
                   "alerts": alerts}, f, ensure_ascii=False, indent=1)
    meta.log_stage("alerts_scan", n_high=sum(1 for a in alerts if a["risk_level"] == "high"),
                   n_mid=sum(1 for a in alerts if a["risk_level"] == "mid"),
                   thresholds=t)
    return alerts


def append_alert_history(alerts: list[dict]) -> None:
    """归档预警历史（F15 预警历史时间线）。"""
    hist_path = get_path("alert_history_file")
    history = []
    if hist_path.exists():
        with open(hist_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    seen = {a["alert_id"] for a in history}
    for a in alerts:
        if a["alert_id"] not in seen:
            history.append(a)
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=1)
