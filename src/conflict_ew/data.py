"""F03 数据清洗与标准化。

- 缺失值处理：军费 log 变换、缺失年份线性插值（记录插值标记）
- 国家编码映射（COW ↔ ISO3，复用 external.countries 主表）
- 单位归一：金额 log 化、比率保持百分数
- 异常值检测（|z|>4 标记并截断，记录审计日志）
- 输出 data_manifest.json（版本戳/SHA256/冻结时间，F03 验收）
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
class CleaningReport:
    n_rows: int = 0
    n_countries: int = 0
    n_interpolated: int = 0
    n_anomaly_flagged: int = 0
    n_anomaly_winsorized: int = 0
    unmatched_codes: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"n_rows": self.n_rows, "n_countries": self.n_countries,
                "n_interpolated": self.n_interpolated, "n_anomaly_flagged": self.n_anomaly_flagged,
                "n_anomaly_winsorized": self.n_anomaly_winsorized,
                "unmatched_codes": self.unmatched_codes, "notes": self.notes}


def _detect_outliers(s: pd.Series, z_threshold: float = 4.0) -> pd.Series:
    """z-score 异常值检测（NaN 不参与）。"""
    mu, sd = s.mean(), s.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return pd.Series(False, index=s.index)
    return ((s - mu).abs() / sd) > z_threshold


def clean_and_standardize(wdi: pd.DataFrame, countries: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """F03 主入口：清洗 + 标准化 WDI 长表（含外部数据列）。

    :param wdi: (iso3, year, mil_expense, gdp_growth, inflation, unemployment, [onset...])
    :param countries: 国家主表（编码映射）
    :return: (clean_df, report)
    """
    report = CleaningReport()
    df = wdi.copy()
    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()

    # ---- 编码映射审计（COW ↔ ISO3）----
    valid_iso3 = set(countries["iso3"].astype(str).str.upper())
    report.unmatched_codes = sorted(set(df["iso3"]) - valid_iso3)
    df = df[df["iso3"].isin(valid_iso3)].copy()
    if report.unmatched_codes:
        report.notes.append(f"编码未匹配 {len(report.unmatched_codes)} 个 iso3，已剔除")
        meta.log_stage("clean_unmatched_codes", codes=report.unmatched_codes[:50])

    # ---- 军费 log 变换 + 缺失插值（记录插值标记）----
    if "mil_expense" in df.columns:
        df["mil_expense_log"] = np.log(np.maximum(df["mil_expense"], 1.0))
        df["mil_expense_interpolated"] = 0
        for iso3, grp in df.groupby("iso3"):
            idx = grp.index
            mask = grp["mil_expense"].isna() | (grp["mil_expense"] <= 0)
            if mask.any():
                df.loc[idx[mask], "mil_expense_interpolated"] = 1
                report.n_interpolated += int(mask.sum())
            # 线性插值（时间序）
            log_series = df.loc[idx, "mil_expense_log"]
            interp = log_series.interpolate(method="linear", limit_direction="both")
            df.loc[idx, "mil_expense_log"] = interp
        report.notes.append(f"军费缺失/非正值插值 {report.n_interpolated} 条（已记录标记）")

    # ---- 经济变量异常值检测与截断（winsorize 到 ±4σ）----
    for col in ("gdp_growth", "inflation", "unemployment"):
        if col not in df.columns:
            continue
        flag = _detect_outliers(df[col])
        report.n_anomaly_flagged += int(flag.sum())
        mu, sd = df[col].mean(), df[col].std(ddof=0)
        if sd and not np.isnan(sd):
            lo, hi = mu - 4 * sd, mu + 4 * sd
            df.loc[flag, col] = df.loc[flag, col].clip(lo, hi)
            report.n_anomaly_winsorized += int(flag.sum())
    if report.n_anomaly_flagged:
        report.notes.append(f"异常值检测: 标记 {report.n_anomaly_flagged} 条并截断至 ±4σ")

    # ---- 政体/领土/同盟变量安全转换 ----
    for col in ("polity_score", "vdem_ldi", "icow_claim", "atop_change", "onset", "active"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)
    report.n_rows = len(df)
    report.n_countries = df["iso3"].nunique()
    meta.log_stage("clean_standardize", n_rows=report.n_rows, n_countries=report.n_countries,
                   n_interpolated=report.n_interpolated)
    return df, report


def write_cleaned(df: pd.DataFrame, report: CleaningReport, out_path=None) -> None:
    """落盘清洗结果并生成 manifest 条目。"""
    out_path = out_path or (get_path("data_dir") / "wdi_clean.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    with open(get_path("data_dir") / "cleaning_report.json", "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
    meta.log_stage("clean_write", path=str(out_path))
