"""01 — Load raw data and write the data manifest + labeled panel."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conflict_ew.config import load_config  # noqa: E402
from conflict_ew.pipeline import seed_everything, stage_load  # noqa: E402

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    seed_everything(int(cfg["reproducibility"]["seed"]))
    out = stage_load(cfg)
    print("manifest:", out["manifest"])
