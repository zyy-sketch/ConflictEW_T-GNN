"""05 — Generate publication-quality vector figures (PDF/SVG + PNG preview)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conflict_ew.config import load_config  # noqa: E402
from conflict_ew.pipeline import seed_everything, stage_figures  # noqa: E402

if __name__ == "__main__":
    cfg = load_config("config.yaml")
    seed_everything(int(cfg["reproducibility"]["seed"]))
    paths = stage_figures(cfg)
    for k, p in paths.items():
        print(f"{k}: {p}")
