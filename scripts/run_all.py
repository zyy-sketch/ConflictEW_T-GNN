"""Run the full pipeline (01 -> 05) end-to-end."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conflict_ew.pipeline import run_all  # noqa: E402

if __name__ == "__main__":
    run_all("config.yaml")
    print("pipeline complete — see artifacts/")
