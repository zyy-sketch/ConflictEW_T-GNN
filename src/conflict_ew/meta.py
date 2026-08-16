"""Runtime provenance capture: versions, seed, git hash, run logs.

Every stage writes a JSONL run-log so the paper's Reproducibility Statement can
reference exact software versions, seed and commit hash.
"""
from __future__ import annotations

import datetime
import importlib
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any


def git_commit(repo_dir: str | Path | None = None) -> str | None:
    """Return the current git commit hash, or None if not in a repo."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(repo_dir) if repo_dir else None,
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def package_version(name: str) -> str | None:
    """Best-effort version lookup for an installed package."""
    try:
        mod = importlib.import_module(name)
        return getattr(mod, "__version__", None)
    except Exception:
        return None


def runtime_info(cfg: dict[str, Any]) -> dict[str, Any]:
    return {
        "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "seed": cfg["reproducibility"]["seed"],
        "git_commit": git_commit(),
        "versions": {
            "numpy": package_version("numpy"),
            "scipy": package_version("scipy"),
            "pandas": package_version("pandas"),
            "sklearn": package_version("sklearn"),
            "xgboost": package_version("xgboost"),
            "shap": package_version("shap"),
            "matplotlib": package_version("matplotlib"),
        },
    }


def write_runlog(cfg: dict[str, Any], stage: str, payload: dict[str, Any]) -> Path:
    """Append a run-log record for `stage` and return the written path."""
    results = Path(cfg["paths"]["results"])
    results.mkdir(parents=True, exist_ok=True)
    rec = {"stage": stage, **runtime_info(cfg), "payload": payload}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    out = results / f"{stage}_{stamp}.json"
    out.write_text(json.dumps(rec, indent=2, ensure_ascii=False), encoding="utf-8")
    return out
