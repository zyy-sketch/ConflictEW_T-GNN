"""Data loading, label construction and manifest generation."""
from __future__ import annotations

import datetime
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def load_raw(cfg: dict[str, Any]) -> dict[str, Any]:
    """Load the dataset into clean structures.

    ``config.data.source`` selects the data origin:
      * ``synthetic`` — the hard-coded ``db_data.json`` (demonstration only).
      * ``real``      — cached real World Bank WDI extract (``data/raw_real.json``);
        if the cache is missing it is fetched automatically by ``data_fetch``.
    """
    base = Path(cfg["data"]["source_dir"])
    source = str(cfg["data"].get("source", "synthetic")).lower()

    if source == "real":
        real_path = base / cfg["data"]["real_file"]
        if not real_path.exists():
            print(f"[data] cache {real_path} missing -> fetching real data automatically")
            from . import data_fetch

            ds = data_fetch.fetch_real_dataset(
                db_path=base / cfg["data"]["full_file"],
                year_min=int(cfg["data"].get("year_min", 1990)),
                year_max=int(cfg["data"].get("year_max", 2024)),
                indicators=cfg["data"].get("wb_indicators", None),
            )
            data_fetch.save_real_dataset(ds, real_path)
        full = real_path
        with open(full, encoding="utf-8") as f:
            d = json.load(f)
        spending_src = d["military_spending"]
        economic_src = d.get("economic", {})
    else:
        full = base / cfg["data"]["full_file"]
        with open(full, encoding="utf-8") as f:
            d = json.load(f)
        spending_src = d["military_spending"]
        economic_src = {}

    countries = pd.DataFrame(d["countries"])
    spending = {code: pd.DataFrame(rows) for code, rows in spending_src.items()}
    economic = {code: pd.DataFrame(rows) for code, rows in economic_src.items()}
    relations = pd.DataFrame(d["relations"])
    conflicts = pd.DataFrame(d["conflicts"])
    factor_scores = dict(d.get("factor_scores", {}))
    alerts = list(d.get("alerts", []))
    settings = dict(d.get("settings", {}))
    tgnn = {code: pd.DataFrame(rows) for code, rows in d.get("tgnn_signals", {}).items()}

    return {
        "countries": countries,
        "spending": spending,
        "economic": economic,
        "relations": relations,
        "conflicts": conflicts,
        "factor_scores": factor_scores,
        "alerts": alerts,
        "settings": settings,
        "tgnn": tgnn,
        "source_path": str(full),
        "source_sha256": _sha256(full),
        "data_source": source,
    }


def build_panel(raw: dict[str, Any]) -> pd.DataFrame:
    """Country-year panel with binary conflict-incidence labels.

    ``label`` = 1 iff the country has a conflict event recorded in that year
    (from ``conflicts``). Features are appended later by ``features.build_features``.
    """
    conflicts = raw["conflicts"]
    conflict_set = set(zip(conflicts["country"], conflicts["year"].astype(int)))
    rows: list[dict[str, Any]] = []
    for code, sp in raw["spending"].items():
        for _, r in sp.iterrows():
            year = int(r["year"])
            rows.append(
                {
                    "country": code,
                    "year": year,
                    "value": float(r["value"]),
                    "label": int((code, year) in conflict_set),
                }
            )
    return pd.DataFrame(rows)


def write_manifest(raw: dict[str, Any], cfg: dict[str, Any], panel: pd.DataFrame) -> dict[str, Any]:
    """Write ``artifacts/data_manifest.json`` and return the manifest dict."""
    art = Path(cfg["paths"]["artifacts"])
    art.mkdir(parents=True, exist_ok=True)
    countries = raw["countries"]
    meta = {
        "source_file": raw["source_path"],
        "data_source": raw.get("data_source", "synthetic"),
        "source_sha256": raw["source_sha256"],
        "snapshot_date": cfg["data"]["snapshot_date"],
        "sipri_note": cfg["data"]["sipri_version"],
        "cow_note": cfg["data"]["cow_version"],
        "countries": int(len(countries)),
        "country_codes": sorted(countries["id"].astype(str).tolist()),
        "spending_series_count": int(len(raw["spending"])),
        "conflict_events": int(len(raw["conflicts"])),
        "relations": int(len(raw["relations"])),
        "panel_rows": int(len(panel)),
        "panel_years": [int(panel["year"].min()), int(panel["year"].max())],
        "positive_labels": int(panel["label"].sum()),
        "generated_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    (art / "data_manifest.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return meta
