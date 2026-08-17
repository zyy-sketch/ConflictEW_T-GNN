"""Real data acquisition from the World Bank World Development Indicators (WDI).

This module lets the pipeline obtain **real** data by itself instead of relying on
the hard-coded synthetic ``db_data.json``. It fetches:

  * Military expenditure (current USD) — indicator ``MS.MIL.XPND.CD``
    (SIPRI data hosted by the World Bank; this is the paper's §3.1 input).
  * GDP growth (annual %)            — ``NY.GDP.MKTP.KD.ZG``
  * Inflation, consumer prices (%)  — ``FP.CPI.TOTL.ZG``
  * Unemployment, total (% labor)   — ``SL.UEM.TOTL.ZS``

The fetched series are cached to ``data/raw_real.json`` so that every downstream
step stays reproducible (offline) and the acquisition is auditable. The conflict
labels / neighbour matrix are reused from ``db_data.json`` (the project's curated
validation set) — swapping in a real Correlates-of-War extract is a documented
next step.

Design notes
------------
* No third-party HTTP client is required: only the standard library ``urllib``.
* A single bulk call per indicator covers all countries (semicolon-separated),
  which keeps the request count low (4 calls total).
* ``countryiso3code`` (not ``country.id``) is used as the join key because it is
  the stable ISO-3 code that matches ``db_data.json`` country ids.
"""
from __future__ import annotations

import datetime
import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

WB_BASE = "https://api.worldbank.org/v2"

# Indicator id -> human label used inside raw_real.json
DEFAULT_INDICATORS: dict[str, str] = {
    "military_usd": "MS.MIL.XPND.CD",
    "gdp_growth": "NY.GDP.MKTP.KD.ZG",
    "inflation": "FP.CPI.TOTL.ZG",
    "unemployment": "SL.UEM.TOTL.ZS",
}


def _fetch_indicator(
    iso3_codes: list[str],
    indicator: str,
    year_min: int,
    year_max: int,
    retries: int = 4,
    timeout: int = 30,
) -> list[dict[str, Any]]:
    """Fetch one indicator for many countries in a single bulk request."""
    codes = ";".join(iso3_codes)
    url = (
        f"{WB_BASE}/country/{codes}/indicator/{indicator}"
        f"?format=json&date={year_min}:{year_max}&per_page=20000"
    )
    last_err = ""
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={
                    "Accept": "application/json",
                    "User-Agent": "ConflictEW/0.2 (research; reproducible pipeline)",
                },
            )
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
            if isinstance(payload, list) and len(payload) >= 2 and isinstance(payload[1], list):
                return payload[1]
            return []
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError) as e:
            last_err = str(e)
            if attempt < retries - 1:
                time.sleep(1.5 * (attempt + 1))
    print(f"  ! indicator {indicator} failed after {retries} attempts: {last_err}")
    return []


def fetch_real_dataset(
    db_path: Path,
    year_min: int = 1990,
    year_max: int = 2024,
    indicators: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build a real-data dict (raw_real.json compatible) from World Bank WDI.

    Returns a dict with keys: ``meta``, ``countries``, ``military_spending``,
    ``economic``, ``relations``, ``conflicts`` (reused from ``db_path``), plus
    empty ``factor_scores``/``alerts``/``settings``/``tgnn`` placeholders so the
    structure is loadable by ``data.load_raw``.
    """
    if indicators is None:
        indicators = DEFAULT_INDICATORS

    with open(db_path, encoding="utf-8") as f:
        db = json.load(f)

    codes: list[str] = [str(c["id"]) for c in db["countries"] if str(c.get("id", "")).strip()]
    codes = sorted(set(codes))
    print(f"[fetch] {len(codes)} countries, years {year_min}-{year_max}")

    # --- bulk fetch each indicator ---
    raw_obs: dict[str, list[dict[str, Any]]] = {}
    for key, ind in indicators.items():
        print(f"[fetch] {key:14s} <- {ind}")
        obs = _fetch_indicator(codes, ind, year_min, year_max)
        raw_obs[key] = obs
        print(f"          got {len(obs)} observations")

    # --- pivot into {country: {year: value}} for each indicator ---
    pivots: dict[str, dict[str, float]] = {k: {} for k in indicators}
    coverage: dict[str, set[str]] = {k: set() for k in indicators}
    for key, obs in raw_obs.items():
        for row in obs:
            code = row.get("countryiso3code") or (row.get("country") or {}).get("id")
            if not code:
                continue
            year = str(row.get("date"))
            val = row.get("value")
            if val is None:
                continue
            try:
                pivots[key][f"{code}|{year}"] = float(val)
                coverage[key].add(code)
            except (TypeError, ValueError):
                continue

    # --- assemble per-country structures ---
    military_spending: dict[str, list[dict[str, Any]]] = {}
    economic: dict[str, list[dict[str, Any]]] = {}
    for code in codes:
        mil = [
            {"year": int(y), "value": pivots["military_usd"][f"{code}|{y}"]}
            for y in (str(yr) for yr in range(year_min, year_max + 1))
            if f"{code}|{y}" in pivots["military_usd"]
        ]
        if mil:
            military_spending[code] = mil
        econ_rows = []
        for yr in range(year_min, year_max + 1):
            y = str(yr)
            g = pivots["gdp_growth"].get(f"{code}|{y}")
            i = pivots["inflation"].get(f"{code}|{y}")
            u = pivots["unemployment"].get(f"{code}|{y}")
            if g is None and i is None and u is None:
                continue
            econ_rows.append(
                {
                    "year": yr,
                    "gdp_growth": g,
                    "inflation": i,
                    "unemployment": u,
                }
            )
        if econ_rows:
            economic[code] = econ_rows

    meta = {
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "source": "World Bank World Development Indicators (api.worldbank.org/v2)",
        "military_expenditure_indicator": indicators["military_usd"],
        "military_expenditure_note": "SIPRI Military Expenditure database, hosted by World Bank WDI",
        "indicators": indicators,
        "year_min": year_min,
        "year_max": year_max,
        "n_countries": len(codes),
        "countries_with_military": len(military_spending),
        "countries_with_economic": len(economic),
        "coverage": {k: sorted(v) for k, v in coverage.items()},
    }

    return {
        "meta": meta,
        "countries": db["countries"],
        "military_spending": military_spending,
        "economic": economic,
        "relations": db["relations"],
        "conflicts": db["conflicts"],
        "factor_scores": {},
        "alerts": [],
        "settings": {},
        "tgnn_signals": {},
    }


def save_real_dataset(dataset: dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dataset, ensure_ascii=False), encoding="utf-8")
    print(f"[fetch] wrote {out_path} ({out_path.stat().st_size // 1024} KB)")
