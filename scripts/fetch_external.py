"""R-8 外部数据源自动抓取、溯源登记与官方原件转换。

三部分功能：
  1. fetch    —— 从官方源下载原始文件到 data/external/raw/（带 UA、超时、SHA-256）
  2. register —— 汇总所有 raw 文件 + 仓库既有 data/external/*.csv 的 SHA-256 与来源
                 → data/external/provenance.json
  3. convert  —— 官方原件 → data/external_real/*.csv（面板 schema），
                 不覆盖 simulate 生成的 data/external/*.csv（保持 golden 复现基准）

用法：
  python scripts/fetch_external.py            # 全流程：fetch + register
  python scripts/fetch_external.py --convert  # 追加官方原件→面板 schema 转换
  python scripts/fetch_external.py --no-fetch # 只重新登记（离线可用）

诚实边界（PROVENANCE.md 同步声明）：
  - data/external/*.csv 是 simulate.write_simulated_data() 的模拟产物（P0-7），
    本脚本不修改它们；真实官方原件经 convert 后落在 data/external_real/。
  - ATOP / ICOW / V-Dem 官方站不可脚本直连（403 / 超时 / 需注册）→ 手动指引。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "external" / "raw"
EXT = ROOT / "data" / "external"
EXT_REAL = ROOT / "data" / "external_real"
UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/120.0 Safari/537.36"
TIMEOUT = 90

# ---------------------------------------------------------------------------
# 源注册表：url / 落盘名 / 解压目标 / 版本 / 手动指引
# ---------------------------------------------------------------------------
SOURCES: dict[str, dict] = {
    "polity5": {
        "name": "Polity5 Annual Time-Series 1946-2018",
        "provider": "Center for Systemic Peace (CSP)",
        "url": "http://www.systemicpeace.org/inscr/p5v2018.xls",
        "dest": "p5v2018.xls",
        "version": "p5v2018 (Polity5d annual, polity2)",
        "manual": None,  # 可自动下载
    },
    "ucdp_intrastate_onset": {
        "name": "UCDP Intrastate Country-Year Conflict Onset",
        "provider": "Uppsala Conflict Data Program (UCDP)",
        "url": "https://ucdp.uu.se/downloads/onset/ucdp-intrastate-country-onset-251.csv",
        "dest": "ucdp-intrastate-country-onset-251.csv",
        "version": "v25.1 (1946-2024, onset1/newconf, gwno_a)",
        "manual": None,
    },
    "cow_intra_state_wars": {
        "name": "COW Intra-State Wars v5.1",
        "provider": "Correlates of War Project",
        "url": "https://correlatesofwar.org/wp-content/uploads/Intra-State-Wars-v5.1.zip",
        "dest": "Intra-State-Wars-v5.1.zip",
        "unzip_dir": "cow_isw",
        "version": "5.1",
        "manual": None,
    },
    "cow_direct_contiguity": {
        "name": "COW Direct Contiguity v3.2",
        "provider": "Correlates of War Project",
        "url": "https://correlatesofwar.org/wp-content/uploads/DirectContiguity320.zip",
        "dest": "DirectContiguity320.zip",
        "unzip_dir": "cow_direct_contiguity",
        "version": "3.2",
        "manual": None,
    },
    "atop": {
        "name": "ATOP Alliance Treaty Obligations and Provisions",
        "provider": "atop-data.org",
        "url": None,
        "manual": "浏览器打开 https://www.atop-data.org/ → Data → 下载 ATOP 5.1 dyadic CSV，"
                  "保存为 data/external/raw/atop_5_1.csv（站点有 Cloudflare 403 防护，无法脚本直连）",
    },
    "icow": {
        "name": "ICOW Territorial Claims",
        "provider": "icow.org (Huth & Allee)",
        "url": None,
        "manual": "浏览器打开 https://www.icow.org/ → Territorial Claims 数据页下载 issue-year 表，"
                  "保存为 data/external/raw/icow_terrclaims.csv（站点当前不可达/超时，需人工核对镜像）",
    },
    "vdem": {
        "name": "V-Dem Country-Year Full Dataset",
        "provider": "v-dem.net",
        "url": None,
        "manual": "浏览器打开 https://www.v-dem.net/ → 注册/登录 → Data → V-Dem Dataset (Country-Year) 下载 CSV，"
                  "保存为 data/external/raw/vdem_cy.csv（需注册许可，禁止脚本直连）",
    },
}


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch_all() -> dict[str, dict]:
    RAW.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict] = {}
    for key, spec in SOURCES.items():
        dest = RAW / spec["dest"] if spec.get("dest") else None
        if spec.get("manual") is None and dest is not None:
            ok = False
            try:
                req = urllib.request.Request(spec["url"], headers={"User-Agent": UA})
                with urllib.request.urlopen(req, timeout=TIMEOUT) as resp, open(dest, "wb") as f:
                    shutil.copyfileobj(resp, f)
                ok = dest.exists() and dest.stat().st_size > 10_000
            except Exception as e:  # noqa: BLE001
                results[key] = {"status": "fail", "error": f"{type(e).__name__}: {e}"}
                continue
            if not ok:
                results[key] = {"status": "fail", "error": "下载内容异常（过小）"}
                continue
            if spec.get("unzip_dir"):
                out_dir = RAW / spec["unzip_dir"]
                out_dir.mkdir(exist_ok=True)
                with zipfile.ZipFile(dest) as z:
                    z.extractall(out_dir)
            results[key] = {
                "status": "ok",
                "url": spec["url"],
                "file": str(dest.relative_to(ROOT)),
                "sha256": sha256_of(dest),
                "bytes": dest.stat().st_size,
            }
        else:
            results[key] = {"status": "manual", "hint": spec["manual"]}
    return results


# ---------------------------------------------------------------------------
# 官方原件 → 面板 schema 转换（写入 data/external_real/，不覆盖 data/external/）
# ---------------------------------------------------------------------------
def _ccode_to_iso3() -> dict[int, str]:
    countries = pd.read_csv(EXT / "countries.csv", dtype=str)
    m: dict[int, str] = {}
    for _, r in countries.iterrows():
        try:
            m[int(float(r["cow_code"]))] = str(r["iso3"]).strip().upper()
        except (ValueError, TypeError):
            continue
    return m


def convert_polity(c2i: dict[int, str], unmatched: set[str]) -> pd.DataFrame:
    src = RAW / "p5v2018.xls"
    df = pd.read_excel(src)
    out = df[df["year"].notna()][["ccode", "scode", "year", "polity2"]].copy()
    # 缺失码 -66/-77/-88 → NaN
    out["polity_score"] = pd.to_numeric(out["polity2"], errors="coerce")
    out.loc[out["polity_score"].isin([-66, -77, -88]), "polity_score"] = None
    # 主映射：Polity5 的 ccode 即 COW 编码 → countries.csv.cow_code；scode 仅作审计
    out["iso3"] = out["ccode"].map(lambda c: c2i.get(int(c)) if pd.notna(c) else None)
    for c in out.loc[out["iso3"].isna(), "ccode"].dropna().unique():
        unmatched.add(f"polity:ccode{int(c)}({df.loc[df['ccode']==c,'scode'].iloc[0] if (df['ccode']==c).any() else '?'})")
    out = out[out["iso3"].notna()]
    out = out[(out["year"] >= 1970) & (out["year"] <= 2024)]
    return out[["iso3", "year", "polity_score"]].sort_values(["iso3", "year"]).reset_index(drop=True)


def convert_cow_conflicts(c2i: dict[int, str], unmatched: set[str]) -> pd.DataFrame:
    """COW Intra-State Wars v5.1 → 国-年标签（历史对照用，非主标签源）。

    数据集覆盖止于 ~2015（max StartYr=2014，closed-war max EndYr=2015）；
    进行中战争 EndYr∈{-7,-8,-9} 以数据集覆盖末端 2015 删失（不丢弃）。
    """
    src = RAW / "cow_isw" / "INTRA-STATE_State_participants v5.1 CSV.csv"
    df = pd.read_csv(src, encoding="latin-1")
    coverage_end = 2015  # v5.1 数据覆盖末端（closed wars 的最大 EndYr）
    rows: list[tuple[str, int, int, int]] = []  # iso3, year, onset, active
    for _, r in df.iterrows():
        for side_ccode in (r["CcodeA"], r["CcodeB"]):
            try:
                ccode = int(float(side_ccode))
            except (ValueError, TypeError):
                continue
            if ccode <= 0:  # -8 非国家行为体
                continue
            iso3 = c2i.get(ccode)
            if iso3 is None:
                unmatched.add(f"cow_conflicts:ccode{ccode}")
                continue
            # 每场战争最多 4 段期间（-8/-9 = 缺失）
            spans = []
            for k in range(1, 5):
                sy, ey = r.get(f"StartYr{k}"), r.get(f"EndYr{k}")
                try:
                    sy = int(float(sy))
                except (ValueError, TypeError):
                    continue
                ey = int(float(ey)) if pd.notna(ey) else -9
                if sy > 0:
                    spans.append((sy, ey))
            if not spans:
                continue
            first_start = min(s for s, _ in spans)
            ends = [e if e > 0 else coverage_end for _, e in spans]
            last_end = max(ends)
            for y in range(last_end, first_start - 1, -1):
                rows.append((iso3, y, int(y == first_start), 1))
    out = pd.DataFrame(rows, columns=["iso3", "year", "onset", "active"])
    g = out.groupby(["iso3", "year"], as_index=False).agg(
        onset=("onset", "max"), active=("active", "max"))
    return g[g["year"] >= 1970].sort_values(["iso3", "year"]).reset_index(drop=True)


def convert_ucdp_labels(c2i: dict[int, str], unmatched: set[str]) -> pd.DataFrame:
    """UCDP Intrastate Country-Year Onset v25.1 → 国-年标签（主标签源）。

    - onset = onset1（UCDP 新冲突首发年：冲突首年或间隔 ≥2 年后重燃）
    - active = conflict_ids 非空（该国当年存在 ≥25 battle deaths 的国家内冲突）
    - gwno_a（Gleditsch-Ward/COW 编码）经 countries.csv.cow_code 映射 ISO3
    """
    src = RAW / "ucdp-intrastate-country-onset-251.csv"
    df = pd.read_csv(src)
    df.columns = [c.strip() for c in df.columns]
    df = df[(df["year"] >= 1970) & (df["year"] <= 2024)]
    df["iso3"] = df["gwno_a"].map(lambda g: c2i.get(int(g)) if pd.notna(g) else None)
    for g in df.loc[df["iso3"].isna(), "gwno_a"].dropna().unique():
        name = df.loc[df["gwno_a"] == g, "name"].iloc[0] if (df["gwno_a"] == g).any() else "?"
        unmatched.add(f"ucdp:gwno{int(g)}({name})")
    out = df[df["iso3"].notna()].copy()
    out["onset"] = pd.to_numeric(out["onset1"], errors="coerce").fillna(0).astype(int)
    out["active"] = out["conflict_ids"].notna().astype(int)
    return out[["iso3", "year", "onset", "active"]].sort_values(["iso3", "year"]).reset_index(drop=True)


def convert_cow_adjacency(c2i: dict[int, str], unmatched: set[str]) -> pd.DataFrame:
    src = RAW / "cow_direct_contiguity" / "DirectContiguity320" / "contdir.csv"
    df = pd.read_csv(src)
    # 面板期 1970-2024 内至少部分重叠的接壤 dyad（conttype>=1）
    rows = set()
    for _, r in df.iterrows():
        if int(r["conttype"]) < 1:
            continue
        begin, end = int(r["begin"]) // 100, int(r["end"]) // 100
        if end < 1970 or begin > 2024:
            continue
        try:
            a, b = c2i.get(int(r["statelno"])), c2i.get(int(r["statehno"]))
        except (ValueError, TypeError):
            continue
        if a is None:
            unmatched.add(f"contiguity:ccode{int(r['statelno'])}")
        if b is None:
            unmatched.add(f"contiguity:ccode{int(r['statehno'])}")
        if a and b and a != b:
            rows.add(tuple(sorted((a, b))))
    return pd.DataFrame(sorted(rows), columns=["iso3_a", "iso3_b"])


def convert_all() -> dict[str, dict]:
    EXT_REAL.mkdir(parents=True, exist_ok=True)
    c2i = _ccode_to_iso3()
    unmatched: set[str] = set()
    report: dict[str, dict] = {}

    try:
        pol = convert_polity(c2i, unmatched)
        pol.to_csv(EXT_REAL / "polity_vdem.csv", index=False)  # 管道 schema 文件名（仅 polity_score，vdem_ldi 待手动）
        report["polity_vdem.csv"] = {"rows": len(pol), "source": "Polity5 p5v2018.xls", "non_null": int(pol["polity_score"].notna().sum())}
    except Exception as e:  # noqa: BLE001
        report["polity_vdem.csv"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        # 主标签源：UCDP（覆盖 1946-2024，支撑 2015-2024 测试窗）
        ucdp = convert_ucdp_labels(c2i, unmatched)
        ucdp.to_csv(EXT_REAL / "cow_conflicts.csv", index=False)  # 管道 schema 文件名
        report["cow_conflicts.csv"] = {"rows": len(ucdp), "source": "UCDP intrastate onset v25.1", "onsets": int(ucdp["onset"].sum()), "active_years": int(ucdp["active"].sum())}
    except Exception as e:  # noqa: BLE001
        report["cow_conflicts.csv"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        # 历史对照：COW ISW v5.1（覆盖止于 2015，不作主标签）
        cow = convert_cow_conflicts(c2i, unmatched)
        cow.to_csv(EXT_REAL / "cow_isw_country_years.csv", index=False)
        report["cow_isw_country_years.csv"] = {"rows": len(cow), "source": "COW Intra-State Wars v5.1", "onsets": int(cow["onset"].sum())}
    except Exception as e:  # noqa: BLE001
        report["cow_isw_country_years.csv"] = {"error": f"{type(e).__name__}: {e}"}

    try:
        adj = convert_cow_adjacency(c2i, unmatched)
        adj.to_csv(EXT_REAL / "cow_adjacency.csv", index=False)
        report["cow_adjacency.csv"] = {"rows": len(adj), "source": "COW Direct Contiguity v3.2 contdir.csv"}
    except Exception as e:  # noqa: BLE001
        report["cow_adjacency.csv"] = {"error": f"{type(e).__name__}: {e}"}

    if unmatched:
        with open(EXT_REAL / "unmatched.log", "w", encoding="utf-8") as f:
            f.write("\n".join(sorted(unmatched)))
        report["_unmatched"] = {"count": len(unmatched), "log": "data/external_real/unmatched.log"}
    return report


# ---------------------------------------------------------------------------
# 溯源登记
# ---------------------------------------------------------------------------
def register(fetch_results: dict[str, dict] | None) -> dict:
    manifest: dict = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "policy": "data/external/*.csv 为 simulate.write_simulated_data() 的模拟产物（P0-7 披露）；"
                  "官方原件在 data/external/raw/，转换产物在 data/external_real/。",
        "sources": {},
        "curated_simulated": {},
        "raw_files": {},
    }
    for key, spec in SOURCES.items():
        entry = {"name": spec["name"], "provider": spec["provider"], "version": spec.get("version")}
        r = (fetch_results or {}).get(key, {})
        entry.update({k: v for k, v in r.items() if k != "name"})
        if r.get("status") != "ok" and spec.get("manual"):
            entry["status"] = entry.get("status", "manual")
            entry["manual_hint"] = spec["manual"]
        manifest["sources"][key] = entry
    for p in sorted(RAW.rglob("*")):
        if p.is_file():
            manifest["raw_files"][str(p.relative_to(ROOT))] = {"sha256": sha256_of(p), "bytes": p.stat().st_size}
    for name in ["cow_conflicts", "polity_vdem", "icow", "atop", "cow_adjacency", "countries"]:
        p = EXT / f"{name}.csv"
        if p.exists():
            manifest["curated_simulated"][f"{name}.csv"] = {
                "sha256": sha256_of(p), "bytes": p.stat().st_size,
                "provenance": "simulate.write_simulated_data()（模拟，非官方数据）",
            }
    out = EXT / "provenance.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return manifest


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--no-fetch", action="store_true", help="跳过下载，仅登记/转换")
    ap.add_argument("--convert", action="store_true", help="官方原件 → data/external_real/ 面板 schema")
    args = ap.parse_args()

    fetch_results = {} if args.no_fetch else fetch_all()
    if args.no_fetch:
        # 离线模式：保留上次 provenance.json 中已登记的下载信息
        prev = EXT / "provenance.json"
        if prev.exists():
            try:
                old = json.loads(prev.read_text(encoding="utf-8")).get("sources", {})
                fetch_results = {k: {kk: vv for kk, vv in v.items() if kk != "manual_hint"}
                                 for k, v in old.items() if k in SOURCES}
            except (json.JSONDecodeError, OSError):
                pass
    print("== fetch ==")
    for k, v in fetch_results.items():
        if v.get("status") == "ok":
            print(f"  [ok]     {k}: {v['bytes']:,} B  sha256={v['sha256'][:12]}…")
        elif v.get("status") == "manual":
            print(f"  [manual] {k}")
        elif v.get("status") == "fail":
            print(f"  [fail]   {k}: {v.get('error')}")

    manifest = register(fetch_results)
    print(f"\n== register ==\n  data/external/provenance.json "
          f"(raw {len(manifest['raw_files'])} files, curated {len(manifest['curated_simulated'])} files)")

    if args.convert:
        print("\n== convert → data/external_real/ ==")
        for name, r in convert_all().items():
            print(f"  {name}: {r}")

    print("\n== manual downloads ==")
    for k, v in fetch_results.items():
        if v.get("status") == "manual" and v.get("hint"):
            print(f"  {k}: {v['hint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
