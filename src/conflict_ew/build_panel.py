"""F04 面板数据构建与存储。

- 构建"国家×年"面板 panel.csv（country, year, 6 因子原始值, y 标签）
- 标签：y_{it}=1 ⟺ 国家 i 在 t 年后 12 个月内进入新冲突 onset（仅首个冲突年记 1）
  （PRD §8.2 标签构建；COW 四标准口径）
- 覆盖 ≥20 年的国家才纳入面板（§8.2）
- 一致性审计：正样本率 ~2-3%、行数 ≈160×55、孤儿行（标签/特征缺失）检查
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from . import meta
from .paths import cfg, get_path


@dataclass
class PanelAudit:
    n_rows: int = 0
    n_countries: int = 0
    n_positive: int = 0
    positive_rate: float = 0.0
    n_excluded_short: int = 0
    excluded_countries: list[str] = field(default_factory=list)
    orphan_rows: int = 0
    issues: list[str] = field(default_factory=list)
    passed: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {"n_rows": self.n_rows, "n_countries": self.n_countries, "n_positive": self.n_positive,
                "positive_rate": round(self.positive_rate, 4), "n_excluded_short": self.n_excluded_short,
                "excluded_countries": self.excluded_countries[:50], "orphan_rows": self.orphan_rows,
                "issues": self.issues, "passed": self.passed}


def build_labels(df: pd.DataFrame, lookahead_months: int = 12) -> pd.DataFrame:
    """按 COW 口径构建 y 标签。

    y_{it}=1 ⟺ 国家 i 在 t 年后 12 个月内进入新冲突 onset；
    仅首个冲突年记 1，后续持续年记 0（冲突进行中不是新 onset）。
    """
    out = df.copy()
    out = out.sort_values(["iso3", "year"]).reset_index(drop=True)
    onset_flags: list[int] = []
    lookback_years = max(1, int(np.ceil(lookahead_months / 12)))
    for iso3, grp in out.groupby("iso3", sort=False):
        onset = grp.set_index("year")["onset"].fillna(0).astype(int)
        for y in grp["year"]:
            window = [onset.get(y + k, 0) for k in range(0, lookback_years + 1)]
            # 未来 12 个月内首个冲突年：以"当年及未来一年出现 onset 且当年未处于持续冲突"计
            y_flag = int(window[0] == 1)  # 当年即 onset（window 各项为 0/1，原恒真条件 sum≥0 已移除）
            if y_flag == 0 and y > onset.index.min():
                # 前瞻：未来一年内有新 onset 且当年不在冲突进行中
                active_now = int(grp.set_index("year").loc[y, "active"]) if y in grp.set_index("year").index else 0
                if active_now == 0 and 1 in window[1:]:
                    # 仅当未来 onset 是"新"事件（前一年未 active）才记 1
                    prev_active = int(grp.set_index("year").loc[y - 1, "active"]) if y - 1 in grp.set_index("year").index else 0
                    y_flag = int(prev_active == 0)
            onset_flags.append(y_flag)
    out["y_onset"] = onset_flags
    return out


def build_panel(clean_df: pd.DataFrame, audit_report_path=None) -> tuple[pd.DataFrame, PanelAudit]:
    """F04 主入口：构建基础面板（原始指标 + y 标签），并做一致性审计。

    :param clean_df: 清洗后的长表（含外部数据列）
    :return: (panel_df, audit)
    """
    pcfg = cfg()["data"]["panel"]
    lcfg = cfg()["data"]["labels"]
    audit = PanelAudit()

    df = clean_df.copy()
    # ---- 覆盖 ≥ min_years 的国家才纳入面板 ----
    coverage = df.groupby("iso3")["year"].nunique()
    short = coverage[coverage < pcfg["min_years"]]
    audit.n_excluded_short = int(len(short))
    audit.excluded_countries = sorted(short.index.tolist())
    if len(short):
        audit.issues.append(f"{len(short)} 国覆盖不足 {pcfg['min_years']} 年，已剔除: {sorted(short.index)[:10]}...")
    df = df[df["iso3"].isin(coverage[coverage >= pcfg["min_years"]].index)].copy()

    # ---- y 标签 ----
    df = build_labels(df, lookahead_months=lcfg["lookahead_months"])

    # ---- 面板列（基础形态；e_star/F1-F6 由 F05/F06 后合并）----
    base_cols = ["iso3", "year", "mil_expense", "mil_expense_log", "mil_expense_interpolated",
                 "gdp_growth", "inflation", "unemployment",
                 "polity_score", "vdem_ldi", "icow_claim", "atop_change",
                 "onset", "active", "y_onset"]
    missing = [c for c in base_cols if c not in df.columns]
    if missing:
        audit.issues.append(f"面板缺失列: {missing}（孤儿行计数按 NaN 处理）")
    panel = df[base_cols].copy() if not missing else df[[c for c in base_cols if c in df.columns]].copy()

    # ---- 一致性审计 ----
    audit.n_rows = len(panel)
    audit.n_countries = panel["iso3"].nunique()
    audit.n_positive = int(panel["y_onset"].sum())
    audit.positive_rate = audit.n_positive / audit.n_rows if audit.n_rows else 0.0
    # 年度连续性：build_labels 的前瞻窗口按年索引（t, t+1），隐含"同一国家年份无缺口"
    gap_countries = int(panel.sort_values(["iso3", "year"]).groupby("iso3")["year"]
                        .apply(lambda s: int((s.diff().dropna() > 1).any())).sum())
    if gap_countries:
        audit.issues.append(
            f"年度不连续国家数: {gap_countries}（存在年份缺口，前瞻标签的窗口语义在该国被截断）")
    target = lcfg["positive_rate_target"]
    if audit.n_positive < 10:
        audit.issues.append(f"正样本过少 ({audit.n_positive})，模型训练不可靠")
        audit.passed = False
    if not (target * 0.5 <= audit.positive_rate <= target * 2.0):
        audit.issues.append(f"正样本率 {audit.positive_rate:.4f} 偏离目标 {target}（范围 {target*0.5:.4f}-{target*2:.4f}）")
    # 孤儿行：y 有值但关键特征缺失
    feats = ["mil_expense_log", "gdp_growth", "inflation", "unemployment"]
    orphan = panel[feats].isna().all(axis=1).sum()
    audit.orphan_rows = int(orphan)
    if orphan:
        audit.issues.append(f"{orphan} 行关键特征全缺失（孤儿行）")
        audit.passed = False

    panel.to_csv(get_path("panel_file"), index=False)
    if audit_report_path is None:
        audit_report_path = get_path("data_dir") / "panel_audit.json"
    audit_report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(audit_report_path, "w", encoding="utf-8") as f:
        json.dump(audit.to_dict(), f, ensure_ascii=False, indent=2)
    meta.log_stage("build_panel", n_rows=audit.n_rows, n_countries=audit.n_countries,
                   n_positive=audit.n_positive, positive_rate=round(audit.positive_rate, 4),
                   audit_passed=audit.passed)
    return panel, audit
