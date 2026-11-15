"""F02 多源数据接入：COW / Polity-IV·V-Dem / ICOW / ATOP 统一 schema 归一化。

- 从 data/external/ 读取外部静态数据集（手动落库或模拟生成），归一化为统一 schema
- 国家编码映射 COW ↔ ISO3（以 countries.csv 为准）
- 编码未匹配的国家写入 unmatched_countries.log（PRD F02 验收）
- 缺失数据源标记 ⛔ 并降级运行（PRD §5.3 / §8.3）
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

import pandas as pd

from . import meta
from .paths import cfg, get_path

logger = logging.getLogger("conflict_ew.external")

# 统一 schema：外部数据集的列名 → 标准列名
SCHEMA_MAP = {
    "cow_conflicts": {"iso3": "iso3", "year": "year", "onset": "onset", "active": "active"},
    "polity_vdem": {"iso3": "iso3", "year": "year", "polity_score": "polity_score", "vdem_ldi": "vdem_ldi"},
    "icow": {"iso3": "iso3", "year": "year", "icow_claim": "icow_claim"},
    "atop": {"iso3": "iso3", "year": "year", "atop_change": "atop_change"},
    "cow_adjacency": {"iso3_a": "iso3_a", "iso3_b": "iso3_b"},
    "countries": None,  # 主表，单独处理
}


@dataclass
class ExternalData:
    """归一化后的外部数据集合。"""
    countries: pd.DataFrame = field(default_factory=pd.DataFrame)          # iso3, cow_code, name_en, name_zh, region
    conflicts: pd.DataFrame = field(default_factory=pd.DataFrame)          # iso3, year, onset, active
    polity: pd.DataFrame = field(default_factory=pd.DataFrame)             # iso3, year, polity_score, vdem_ldi
    icow: pd.DataFrame = field(default_factory=pd.DataFrame)               # iso3, year, icow_claim
    atop: pd.DataFrame = field(default_factory=pd.DataFrame)               # iso3, year, atop_change
    adjacency: pd.DataFrame = field(default_factory=pd.DataFrame)          # iso3_a, iso3_b
    source_status: dict[str, dict] = field(default_factory=dict)           # 数据源状态（✅/⛔）
    unmatched: list[dict] = field(default_factory=list)                    # 编码未匹配国家


def _iso3_lookup(countries: pd.DataFrame) -> dict[str, str]:
    """构建 iso3 -> cow_code 映射（str 化 cow_code）。"""
    m: dict[str, str] = {}
    for _, r in countries.iterrows():
        m[str(r["iso3"]).strip().upper()] = str(r.get("cow_code", "")).strip()
    return m


def _read_csv_or_none(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None
    return pd.read_csv(path, dtype=str)


def _normalize_frame(df: pd.DataFrame, col_map: dict[str, str]) -> pd.DataFrame:
    """按 col_map 重命名并保留标准列。"""
    rename = {k: v for k, v in col_map.items() if k in df.columns}
    out = df.rename(columns=rename)
    keep = [v for v in col_map.values() if v in out.columns]
    return out[keep] if keep else out


def active_external_dir() -> Path:
    """按 config.data.external_source 解析当前生效的外部数据目录。

    - "simulated"（默认）：data/external —— simulate.py 生成/管理的目录
    - "real"：paths.external_dir_real（data/external_real）—— 官方原件转换产物
    """
    source = (cfg().get("data", {}) or {}).get("external_source", "simulated")
    if source == "real" and cfg()["paths"].get("external_dir_real"):
        return get_path("external_dir_real")
    return get_path("external_dir")


def ingest_external(ext_dir: Optional[Path] = None, write_unmatched: bool = True) -> ExternalData:
    """F02 主入口：读取并归一化全部外部数据集。

    :param ext_dir: 外部数据目录（默认按 data.external_source 解析）
    :param write_unmatched: 是否将未匹配国家写入 unmatched_countries.log
    """
    ext_dir = ext_dir or active_external_dir()
    ext_dir.mkdir(parents=True, exist_ok=True)
    data = ExternalData()
    source_status: dict[str, dict] = {}

    # ---- 主表 countries（编码映射基准）----
    countries = _read_csv_or_none(ext_dir / "countries.csv")
    if countries is not None:
        data.countries = _normalize_frame(countries, {"iso3": "iso3", "cow_code": "cow_code",
                                                      "name_en": "name_en", "name_zh": "name_zh", "region": "region"})
        source_status["countries"] = {"status": "✅", "rows": len(data.countries), "version": "simulated-or-manual"}
    else:
        from .simulate import COUNTRIES

        data.countries = pd.DataFrame(COUNTRIES)
        source_status["countries"] = {"status": "⛔", "rows": 0, "note": "countries.csv 缺失，使用内置国家表"}

    iso3_to_cow = _iso3_lookup(data.countries)

    # ---- 各外部数据集 ----
    for src, col_map in SCHEMA_MAP.items():
        if src == "countries":
            continue
        df = _read_csv_or_none(ext_dir / f"{src}.csv")
        if df is None:
            source_status[src] = {"status": "⛔", "rows": 0, "note": "未落库（降级运行）"}
            continue
        if col_map is None:
            continue
        norm = _normalize_frame(df, col_map)
        # 数值化
        num_cols = [c for c in norm.columns if c not in ("iso3", "iso3_a", "iso3_b", "cow_code")]
        for c in num_cols:
            norm[c] = pd.to_numeric(norm[c], errors="coerce")
        if "iso3" in norm.columns:
            norm["iso3"] = norm["iso3"].str.strip().str.upper()
        if "year" in norm.columns:
            norm["year"] = pd.to_numeric(norm["year"], errors="coerce").astype("Int64")
        # 编码匹配审计（仅对含 iso3 列的数据集；邻接矩阵用 iso3_a/iso3_b）
        unmatched = []
        if "iso3" in norm.columns:
            for iso3 in set(norm["iso3"].dropna().astype(str)):
                if iso3 not in iso3_to_cow:
                    unmatched.append({"dataset": src, "iso3": iso3})
            data.unmatched.extend(unmatched)
        setattr(data, src.replace("cow_", "").replace("polity_vdem", "polity"), norm)
        source_status[src] = {"status": "✅" if not unmatched else "⚠️",
                              "rows": len(norm),
                              "unmatched": len(unmatched),
                              "version": "simulated-or-manual",
                              "iso3_to_cow": "✅" if iso3_to_cow else "⛔"}
    data.source_status = source_status

    if write_unmatched and data.unmatched:
        log_path = ext_dir / "unmatched_countries.log"
        with open(log_path, "w", encoding="utf-8") as f:
            for u in data.unmatched:
                f.write(json.dumps(u, ensure_ascii=False) + "\n")
        meta.log_stage("external_unmatched", n=len(data.unmatched), path=str(log_path))
    meta.log_stage("external_ingest",
                   statuses={k: v.get("status") for k, v in source_status.items()},
                   unmatched=len(data.unmatched))
    return data


def merge_external_to_wdi(wdi: pd.DataFrame, ext: ExternalData) -> pd.DataFrame:
    """将外部静态数据（COW 标签/Polity/ICOW/ATOP）并入 WDI 长表。

    以 wdi 的 (iso3, year) 为基准左连接；缺失外部数据源时相应列置 NaN/0 并降级。
    """
    df = wdi.copy()
    joins = [
        ("conflicts", ["onset", "active"], "COW 冲突标签"),
        ("polity", ["polity_score", "vdem_ldi"], "Polity/V-Dem"),
        ("icow", ["icow_claim"], "ICOW 领土主张"),
        ("atop", ["atop_change"], "ATOP 同盟条约"),
    ]
    for attr, cols, label in joins:
        tbl = getattr(ext, attr, None)
        if tbl is not None and len(tbl) > 0:
            df = df.merge(tbl, on=["iso3", "year"], how="left", suffixes=("", f"_{attr}"))
            for c in cols:
                if c not in df.columns:
                    df[c] = 0
            # 真实来源为稀疏表（仅事件年有行）：标签列缺失年份必须显式置 0
            if attr == "conflicts":
                df[["onset", "active"]] = df[["onset", "active"]].fillna(0)
        else:
            for c in cols:
                df[c] = 0
            meta.log_stage("external_missing", source=label, degraded=True)
    # 邻接矩阵单独保存（factors.py 使用）
    if len(ext.adjacency) > 0:
        adj_path = active_external_dir() / "cow_adjacency_normalized.csv"
        ext.adjacency.to_csv(adj_path, index=False)
    return df
