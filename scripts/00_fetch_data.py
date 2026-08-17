"""00 — Acquire real data from the World Bank WDI.

Replaces the hard-coded synthetic ``db_data.json`` military series with real
SIPRI military-expenditure data plus macro indicators, cached to
``data/raw_real.json``. The rest of the pipeline (Kalman -> multi-factor ->
XGBoost -> eval) then runs on real inputs via ``config.data.source: real``.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conflict_ew.config import load_config  # noqa: E402
from conflict_ew.data_fetch import fetch_real_dataset, save_real_dataset  # noqa: E402

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    dcfg = cfg["data"]
    db_path = Path(dcfg["source_dir"]) / dcfg["full_file"]
    out_path = Path(dcfg["source_dir"]) / dcfg["real_file"]
    year_min = int(dcfg.get("year_min", 1990))
    year_max = int(dcfg.get("year_max", 2024))
    indicators = dcfg.get("wb_indicators", None)

    print("=== 00_fetch_data: acquiring real World Bank WDI data ===")
    ds = fetch_real_dataset(db_path, year_min, year_max, indicators)
    save_real_dataset(ds, out_path)
    m = ds["meta"]
    print("=== done ===")
    print(f"  countries requested : {m['n_countries']}")
    print(f"  with military data  : {m['countries_with_military']}")
    print(f"  with economic data  : {m['countries_with_economic']}")
