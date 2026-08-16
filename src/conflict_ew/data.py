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
    """Load the full-field dataset (``db_data.json``) into clean structures."""
    base = Path(cfg["data"]["source_dir"])
    full = base / cfg["data"]["full_file"]
    with open(full, encoding="utf-8") as f:
        d = json.load(f)

    countries = pd.DataFrame(d["countries"])
    spending = {code: pd.DataFrame(rows) for code, rows in d["military_spending"].items()}
    relations = pd.DataFrame(d["relations"])
    conflicts = pd.DataFrame(d["conflicts"])
    factor_scores = dict(d["factor_scores"])
    alerts = list(d["alerts"])
    settings = dict(d["settings"])
    tgnn = {code: pd.DataFrame(rows) for code, rows in d.get("tgnn_signals", {}).items()}

    return {
        "countries": countries,
        "spending": spending,
        "relations": relations,
        "conflicts": conflicts,
        "factor_scores": factor_scores,
        "alerts": alerts,
        "settings": settings,
        "tgnn": tgnn,
        "source_path": str(full),
        "source_sha256": _sha256(full),
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
