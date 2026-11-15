# -*- coding: utf-8 -*-
"""WDI ↔ SIPRI 军费口径抽核（决策 A：数据源保持 WDI，论文侧改口的实证支撑）。

WDI `MS.MIL.XPND.CD`（现价美元）与 SIPRI Military Expenditure Database
"Current US$" 表（百万现价美元）应为同口径转发。本脚本：
1. 下载（或复用缓存）SIPRI 官方 xlsx，登记溯源（URL/版本/修订日/SHA-256/DOI）；
2. 抽 20 国 × 5 个年份，逐点比对相对差；
3. 输出一致率 → 支撑论文 TABLE IV/Data Availability 的口径描述。

输出：artifacts/wdi_sipri_check.json
"""
from __future__ import annotations

import difflib
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

OUT = ROOT / "artifacts"
RAW = ROOT / "data" / "external" / "raw"

SIPRI_URL = "https://www.sipri.org/sites/default/files/SIPRI-Milex-data-1949-2025_v1.2.xlsx"
SIPRI_VERSION = "v1.2 (revised 2026-04-27; DOI 10.55163/CQGC9685)"
SAMPLE_ISO = ["USA", "CHN", "RUS", "GBR", "DEU", "FRA", "JPN", "IND", "BRA", "TUR",
              "ISR", "SAU", "KOR", "CAN", "AUS", "POL", "NOR", "NGA", "ZAF", "ARG"]
SAMPLE_YEARS = [2000, 2010, 2015, 2020, 2023]
# SIPRI 国名 → ISO3（别名兜底；SIPRI 亦用 "USA" 等短名）
ALIAS = {"United States of America": "USA", "United States": "USA", "USA": "USA",
         "United Kingdom": "GBR", "UK": "GBR", "Russia": "RUS", "Korea, South": "KOR",
         "South Korea": "KOR", "Türkiye": "TUR", "Turkey": "TUR"}


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_sipri_file() -> Path:
    RAW.mkdir(parents=True, exist_ok=True)
    dest = RAW / "SIPRI-Milex-data-1949-2025_v1.2.xlsx"
    if dest.exists() and dest.stat().st_size > 500_000:
        return dest
    r = requests.get(SIPRI_URL, timeout=180, headers={"User-Agent": "Mozilla/5.0"})
    r.raise_for_status()
    dest.write_bytes(r.content)
    return dest


def load_sipri_current_usd(path: Path) -> dict[str, dict[int, float]]:
    """解析 'Current US$' 表 → {国名: {年: 百万美元}}。"""
    from openpyxl import load_workbook
    wb = load_workbook(path, read_only=True, data_only=True)
    ws = wb["Current US$"]
    header_year = None
    out: dict[str, dict[int, float]] = {}
    for row in ws.iter_rows(values_only=True):
        cells = list(row)
        if header_year is None:
            if cells and str(cells[0]).strip() == "Country":
                header_year = [int(c) if c is not None else None for c in cells[2:]]
            continue
        name = str(cells[0]).strip() if cells[0] else None
        if not name or name in ("Notes",):
            continue
        vals = {}
        for yr, v in zip(header_year, cells[2:]):
            if yr is None or v is None:
                continue
            s = str(v).strip()
            if s in ("xxx", ". .", "", "…"):
                continue
            try:
                vals[int(yr)] = float(v)
            except ValueError:
                continue
        if vals:
            out[name] = vals
    return out


def main() -> None:
    f = ensure_sipri_file()
    provenance = {"source": "SIPRI Military Expenditure Database", "url": SIPRI_URL,
                  "version": SIPRI_VERSION, "downloaded_at": date.today().isoformat(),
                  "sha256": sha256_of(f), "bytes": f.stat().st_size,
                  "license": "CC BY 4.0（商业使用需 SIPRI 授权）；用于口径核验",
                  "wdi_series": "MS.MIL.XPND.CD (current US$), snapshot 2024-06-20"}

    sipri = load_sipri_current_usd(f)
    # 名称 → ISO3：别名表 + countries.csv 的 name_en + difflib 兜底
    cname = pd.read_csv(ROOT / "data" / "external" / "countries.csv")
    name2iso = dict(zip(cname["name_en"], cname["iso3"]))
    name2iso.update(ALIAS)
    wdi = pd.DataFrame(json.loads((ROOT / "data" / "raw_real.json").read_text(encoding="utf-8"))["records"])
    wdi_lookup = {(r["iso3"], int(r["year"])): float(r["mil_expense"])
                  for _, r in wdi.iterrows() if pd.notna(r["mil_expense"])}

    comparisons = []
    unresolved = []
    for iso in SAMPLE_ISO:
        # SIPRI 名匹配
        if iso in set(name2iso.values()):
            sipri_name = next((nm for nm, i in name2iso.items() if i == iso and nm in sipri), None)
        if sipri_name is None:
            cand = difflib.get_close_matches(iso, list(sipri), n=1)
            sipri_name = cand[0] if cand else None
        if sipri_name is None:
            unresolved.append(iso)
            continue
        for yr in SAMPLE_YEARS:
            sv = sipri[sipri_name].get(yr)
            wv = wdi_lookup.get((iso, yr))
            if sv is None or sv is None or wv is None or not np.isfinite(wv):
                comparisons.append({"iso3": iso, "year": yr, "sipri_musd": sv,
                                    "wdi_usd": wv, "rel_diff": None, "note": "missing"})
                continue
            rel = abs(wv - sv * 1e6) / abs(sv * 1e6)
            comparisons.append({"iso3": iso, "year": yr, "sipri_musd": sv,
                                "wdi_usd": wv, "rel_diff": round(float(rel), 6)})
    df = pd.DataFrame(comparisons)
    ok = df.dropna(subset=["rel_diff"])
    res = {"provenance": provenance,
           "n_compared": int(len(ok)),
           "n_missing": int(df["rel_diff"].isna().sum()),
           "unresolved_countries": unresolved,
           "share_within_1pct": round(float((ok["rel_diff"] < 0.01).mean()), 4),
           "share_within_0.1pct": round(float((ok["rel_diff"] < 0.001).mean()), 4),
           "median_rel_diff": round(float(ok["rel_diff"].median()), 6),
           "max_rel_diff": round(float(ok["rel_diff"].max()), 6),
           "comparisons": comparisons,
           "conclusion": "若一致率高（>90% country-years <1%），支持论文口径描述："
                         "WDI MS.MIL.XPND.CD 即 SIPRI 口径（现价美元）转发；"
                         "差异来自 SIPRI 年度修订版本与快照日期。"}
    print(json.dumps({k: v for k, v in res.items() if k != "comparisons"},
                     ensure_ascii=False, indent=1))
    (OUT / "wdi_sipri_check.json").write_text(json.dumps(res, ensure_ascii=False, indent=2),
                                              encoding="utf-8")
    print("DONE: artifacts/wdi_sipri_check.json")


if __name__ == "__main__":
    main()
