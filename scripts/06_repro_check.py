"""06 — Reproducibility self-check: re-run twice, compare artifact hashes.

A metric pipeline is reproducible only if identical seeds produce bit-identical
results. This script re-runs the train/eval stages with a fixed seed and diffs
the hashes of the key artifacts.
"""
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conflict_ew.config import load_config  # noqa: E402
from conflict_ew.pipeline import seed_everything, stage_features, stage_train, stage_eval  # noqa: E402


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    cfg = load_config("config.yaml")
    art = Path(cfg["paths"]["artifacts"])
    targets = ["features.csv", "kalman_params.csv", "oos_predictions.csv", "metrics.json"]

    hashes = []
    for run in (1, 2):
        seed_everything(int(cfg["reproducibility"]["seed"]))
        stage_features(cfg)
        stage_train(cfg)
        stage_eval(cfg)
        hashes.append({t: _hash(art / t) for t in targets})

    ok = True
    for t in targets:
        same = hashes[0][t] == hashes[1][t]
        ok &= same
        print(f"{t}: {'MATCH' if same else 'DIFFER'}")

    print("\nREPRODUCIBILITY:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)
