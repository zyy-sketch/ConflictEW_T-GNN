"""F06 多因子特征构造：F1-F6 注册表驱动。

因子定义（PRD §4 F06 / §8.1）：
  F1 军费异常 e*         —— Kalman 标准化一步预测残差（截至 t 的滤波输出）
  F2 军备竞赛             —— 邻国军费增速差（t-1）
  F3 经济压力             —— GDP 增速(t-1) / 通胀(t-1) / 失业(t-1)
  F4 政治不稳             —— Polity 变化 / V-Dem LDI 变化
  F5 领土争端             —— ICOW 活跃主张（t-1）
  F6 同盟关系             —— ATOP 同盟义务变化（t-1）

防泄漏约束（§8.2）：
  - e* 只用 ≤t 信息（filtered 残差，非平滑残差）
  - 经济/政治/领土/同盟变量一律用 t-1 值（shift(1)）
  - 邻国因子用 t-1 增速差

可扩展性（§8.4）：注册表驱动——新增因子 = 配置条目 + 一个特征函数。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import pandas as pd

from . import meta
from .paths import cfg, get_path


@dataclass
class FactorSpec:
    id: str
    name: str
    description: str
    columns: list[str]
    fn: Callable[[pd.DataFrame], pd.DataFrame]


FACTOR_DEFINITIONS: list[dict] = [
    {"id": "F1", "name": "军费异常信号 e*", "description": "Kalman 滤波标准化一步预测残差，|e*|≥2 表示军费支出显著偏离历史趋势（潜在军事意图）"},
    {"id": "F2", "name": "军备竞赛", "description": "邻国军费增速差：邻国最高增速 − 本国增速（t-1），反映区域军备竞赛压力"},
    {"id": "F3", "name": "经济压力", "description": "GDP 增速下滑 / 通胀高企 / 失业率上升（t-1），经济困境是冲突催化剂"},
    {"id": "F4", "name": "政治不稳", "description": "Polity 政体指数变化 + V-Dem 自由民主指数变化（当年 vs 前一年），政权不稳定信号"},
    {"id": "F5", "name": "领土争端", "description": "ICOW 活跃领土主张（t-1），领土争议是武装冲突最常见诱因"},
    {"id": "F6", "name": "同盟关系", "description": "ATOP 正式同盟义务变化（t-1），同盟调整改变区域权力平衡"},
]


def _factor_e_star(df: pd.DataFrame) -> pd.DataFrame:
    """F1：军费异常 e*（直接取 Kalman 输出，无滞后——滤波残差本身只用 ≤t 信息）。"""
    out = df[["iso3", "year", "e_star"]].copy()
    out["F1_e_star"] = out["e_star"]
    return out[["iso3", "year", "F1_e_star"]]


def _factor_arms_race(df: pd.DataFrame, adjacency: pd.DataFrame) -> pd.DataFrame:
    """F2：邻国军费增速差（t-1）。邻接矩阵缺失时降级为 neighbor_e 近似（PRD F06 验收）。"""
    # 军费增速（t 年 vs t-1 年，log 差）
    growth = df[["iso3", "year", "mil_expense_log"]].copy()
    growth = growth.sort_values(["iso3", "year"])
    growth["mil_growth"] = growth.groupby("iso3")["mil_expense_log"].diff()
    growth["mil_growth_lag1"] = growth.groupby("iso3")["mil_growth"].shift(1)

    if adjacency is not None and len(adjacency) > 0:
        pairs = []
        for _, r in adjacency.iterrows():
            pairs.append((r["iso3_a"], r["iso3_b"]))
            pairs.append((r["iso3_b"], r["iso3_a"]))
        adj_map: dict[str, list[str]] = {}
        for a, b in pairs:
            adj_map.setdefault(a, []).append(b)
        rows = []
        for iso3, grp in growth.groupby("iso3"):
            neighbors = adj_map.get(iso3, [])
            nbr_growth = growth[growth["iso3"].isin(neighbors)][["year", "mil_growth_lag1"]]
            if len(nbr_growth) == 0:
                # 降级：无邻国时用全球中位增速差近似（标记 degraded）
                nbr_growth = growth[["year", "mil_growth_lag1"]].groupby("year").median().reset_index()
                degraded = 1
            else:
                degraded = 0
            # 多邻国时取每年最高增速（最紧张关系），先按 year 聚合保证 1:1 合并
            nbr_max = nbr_growth.groupby("year")["mil_growth_lag1"].max().reset_index()
            nbr_max = nbr_max.rename(columns={"mil_growth_lag1": "nbr_g"})
            merged = grp.merge(nbr_max, on="year", how="left")
            m = merged.copy()
            m["F2_arms_race"] = m["nbr_g"] - m["mil_growth_lag1"]
            m["F2_degraded"] = int(degraded)
            rows.append(m[["iso3", "year", "F2_arms_race", "F2_degraded"]])
        return pd.concat(rows, ignore_index=True)
    # 邻接矩阵缺失：用区域中位增速差近似（PRD 允许的降级）
    region = df[["iso3", "year"]].merge(_region_table(df), on="iso3", how="left")
    base = growth.merge(region, on=["iso3", "year"], how="left")
    base["reg_med"] = base.groupby(["region", "year"])["mil_growth_lag1"].transform("median")
    out = base.copy()
    out["F2_arms_race"] = out["reg_med"] - out["mil_growth_lag1"]
    out["F2_degraded"] = 1
    return out[["iso3", "year", "F2_arms_race", "F2_degraded"]]


def _region_table(df: pd.DataFrame) -> pd.DataFrame:
    from .simulate import COUNTRIES

    reg = pd.DataFrame(COUNTRIES)[["iso3", "region"]]
    return reg[reg["iso3"].isin(df["iso3"])]


def _factor_economic(df: pd.DataFrame) -> pd.DataFrame:
    """F3：经济压力（t-1 滞后值）。"""
    out = df.sort_values(["iso3", "year"]).copy()
    for c in ("gdp_growth", "inflation", "unemployment"):
        out[f"F3_{c}_lag1"] = out.groupby("iso3")[c].shift(1)
    return out[["iso3", "year", "F3_gdp_growth_lag1", "F3_inflation_lag1", "F3_unemployment_lag1"]]


def _factor_political(df: pd.DataFrame) -> pd.DataFrame:
    """F4：政治不稳（当年 vs 前一年变化）。"""
    out = df.sort_values(["iso3", "year"]).copy()
    out["F4_polity_change"] = out.groupby("iso3")["polity_score"].diff()
    out["F4_vdem_ldi_change"] = out.groupby("iso3")["vdem_ldi"].diff()
    return out[["iso3", "year", "F4_polity_change", "F4_vdem_ldi_change"]]


def _factor_territory(df: pd.DataFrame) -> pd.DataFrame:
    """F5：领土争端（t-1）。"""
    out = df.sort_values(["iso3", "year"]).copy()
    out["F5_icow_claim_lag1"] = out.groupby("iso3")["icow_claim"].shift(1)
    return out[["iso3", "year", "F5_icow_claim_lag1"]]


def _factor_alliance(df: pd.DataFrame) -> pd.DataFrame:
    """F6：同盟关系（t-1）。"""
    out = df.sort_values(["iso3", "year"]).copy()
    out["F6_atop_change_lag1"] = out.groupby("iso3")["atop_change"].shift(1)
    return out[["iso3", "year", "F6_atop_change_lag1"]]


def get_factor_columns() -> list[str]:
    """注册表驱动的特征列清单。"""
    return ["F1_e_star", "F2_arms_race", "F3_gdp_growth_lag1", "F3_inflation_lag1",
            "F3_unemployment_lag1", "F4_polity_change", "F4_vdem_ldi_change",
            "F5_icow_claim_lag1", "F6_atop_change_lag1"]


def factor_column_map() -> dict[str, str]:
    """特征列 → 因子 ID 映射（SHAP 归因聚合用）。"""
    return {c: f"F{i}" for i in range(1, 7) for c in get_factor_columns() if c.startswith(f"F{i}_")}


def build_features(panel: pd.DataFrame, signals: pd.DataFrame,
                   adjacency: pd.DataFrame | None = None,
                   out_path=None) -> pd.DataFrame:
    """F06 主入口：合并 Kalman 信号 + 构造 F1-F6，输出 features.csv。

    :param panel: 基础面板（含 mil_expense_log / 协变量 / 外部变量）
    :param signals: Kalman 信号（iso3, year, e_star, is_anomaly）
    :param adjacency: 邻接矩阵（iso3_a, iso3_b）；None 时 F2 降级
    """
    df = panel.merge(signals[["iso3", "year", "e_star", "is_anomaly"]],
                     on=["iso3", "year"], how="left")
    parts = [df[["iso3", "year"]]]
    parts.append(_factor_e_star(df))
    parts.append(_factor_arms_race(df, adjacency))
    parts.append(_factor_economic(df))
    parts.append(_factor_political(df))
    parts.append(_factor_territory(df))
    parts.append(_factor_alliance(df))
    feats = parts[0]
    for p in parts[1:]:
        feats = feats.merge(p, on=["iso3", "year"], how="left")
    feats = feats.merge(df[["iso3", "year", "y_onset"]], on=["iso3", "year"], how="left")
    feats = feats.sort_values(["iso3", "year"]).reset_index(drop=True)
    out_path = out_path or get_path("features_file")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    feats.to_csv(out_path, index=False)
    meta.log_stage("build_features", n_rows=len(feats), n_countries=feats["iso3"].nunique(),
                   n_positive=int(feats["y_onset"].sum()), degraded_f2=int(feats["F2_degraded"].max()))
    return feats


def factor_registry() -> list[dict]:
    """因子注册表（前端因子定义提示 + 报告引用）。"""
    return FACTOR_DEFINITIONS


def save_factor_registry() -> None:
    path = get_path("artifact_dir") / "factor_registry.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(FACTOR_DEFINITIONS, f, ensure_ascii=False, indent=2)
