"""元数据与可复现性基础设施（对应 PRD §8.2 / F03 / 11.1）。

- 固定种子 20240101（random / numpy / xgboost 统一）
- 逐阶段 JSONL run-log（git hash / Python 与库版本 / 种子 / 时间戳）
- data_manifest.json 版本戳 + SHA256（F03/F04）
- 06_repro_check.py 依赖的确定性哈希
"""
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .paths import cfg, get_path

SEED = 20240101


def set_seed(seed: int = SEED) -> int:
    """统一设置随机种子（random/numpy/xgboost）。返回实际种子。"""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import xgboost as xgb

        xgb.set_config(verbosity=0)
    except Exception:  # pragma: no cover - xgboost 未安装时忽略
        pass
    # 极端受限环境（只读 environ）下赋值可能失败；静默忽略不影响 random/numpy 种子
    try:
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:  # noqa: BLE001  # 防御性降级，见上注释
        pass
    return seed


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_hash() -> str:
    """当前 git 提交 hash；非 git 仓库时返回 'nogit'。"""
    try:
        r = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=5, check=False,
            cwd=str(get_path("data_dir").parent),
        )
        if r.returncode == 0:
            return r.stdout.strip()
    except Exception:  # noqa: BLE001  # git 缺失/超时等一律降级为 nogit（run-log 用，非关键路径）
        pass
    return "nogit"


def env_versions() -> dict[str, str]:
    """Python 与关键库版本，写入 run-log 保证可复现。"""
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod_name in ("numpy", "pandas", "scipy", "sklearn", "xgboost", "shap", "yaml", "requests"):
        try:
            mod = __import__(mod_name)
            versions[mod_name] = getattr(mod, "__version__", "unknown")
        except Exception:  # noqa: BLE001  # 任一依赖缺失记录 MISSING，不中断 manifest
            versions[mod_name] = "MISSING"
    return versions


# ---------------------------------------------------------------- run-log
_run_log: Optional[list[dict]] = None
_run_log_path: Optional[Path] = None


def init_run_log() -> Path:
    """初始化 run-log（JSONL），记录运行元信息。返回日志路径。"""
    global _run_log, _run_log_path
    _run_log_path = get_path("run_log_file")
    _run_log_path.parent.mkdir(parents=True, exist_ok=True)
    # 覆盖清空（不使用 unlink，兼容受限沙箱环境），保证每次运行可两跑对比
    with open(_run_log_path, "w", encoding="utf-8") as f:
        f.write("")
    _run_log = []
    log_stage(
        "init",
        git_hash=git_hash(),
        seed=SEED,
        versions=env_versions(),
        timestamp=utcnow_iso(),
    )
    return _run_log_path


def log_stage(stage: str, **kwargs: Any) -> None:
    """追加一条阶段记录并落盘。"""
    record = {"stage": stage, "timestamp": utcnow_iso(), **kwargs}
    if _run_log is None:
        # 未显式 init 时自动初始化
        init_run_log()
    _run_log.append(record)
    with open(_run_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")


def load_run_log() -> list[dict]:
    path = get_path("run_log_file")
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


# ---------------------------------------------------------------- manifest
def sha256_file(path: Path | str) -> str:
    """计算文件 SHA256（F03 一致性审计）。"""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _manifest_entry(source: str, path: Path, version: str = "1.0") -> dict:
    if path.exists():
        return {
            "source": source,
            "version": version,
            "path": str(path.relative_to(get_path("data_dir").parent)),
            "sha256": sha256_file(path),
            "size_bytes": path.stat().st_size,
            "frozen_at": utcnow_iso(),
        }
    return {
        "source": source,
        "version": version,
        "path": str(path),
        "sha256": None,
        "size_bytes": None,
        "frozen_at": utcnow_iso(),
        "status": "missing",
    }


def write_manifest(extra: Optional[list[dict]] = None, **meta: Any) -> dict:
    """生成 data_manifest.json（F03：版本戳/SHA256/冻结时间）。"""
    files = [
        get_path("raw_real_file"),
        get_path("simulated_data_file"),
        get_path("panel_file"),
        get_path("features_file"),
        get_path("kalman_signals_file"),
        get_path("oos_file"),
        get_path("shap_file"),
        get_path("metrics_file"),
    ]
    entries = {}
    for i, p in enumerate(files):
        src = p.stem if p.exists() else p.name
        entries[src] = _manifest_entry(src, p)
    from .external import active_external_dir

    d = active_external_dir()
    if d.exists():
        for f in sorted(d.glob("*.csv")):
            entries[f.stem] = _manifest_entry(f.stem, f)
    if extra:
        for e in extra:
            entries[e.get("source", e.get("path", "extra"))] = e
    manifest = {
        "project": cfg()["project"]["name"],
        "version": cfg()["project"]["version"],
        "seed": SEED,
        "generated_at": utcnow_iso(),
        "git_hash": git_hash(),
        "frozen": True,
        "entries": entries,
        **meta,
    }
    mf_path = get_path("manifest_file")
    mf_path.parent.mkdir(parents=True, exist_ok=True)
    with open(mf_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    log_stage("manifest", sha256=sha256_file(mf_path))
    return manifest


def load_manifest() -> dict:
    mf_path = get_path("manifest_file")
    if not mf_path.exists():
        return {}
    return json.loads(mf_path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------- 确定性哈希（06_repro_check.py 使用）
def hash_of_json(data: Any) -> str:
    """对任意 JSON 序列化数据计算确定性 SHA256（key 排序）。"""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()


def pipeline_hash() -> str:
    """汇总全部核心产物的确定性哈希（两跑一致即复现成功）。"""
    parts: dict[str, Any] = {"seed": SEED}
    for key in ("panel_file", "features_file", "kalman_signals_file", "oos_file", "shap_file", "metrics_file"):
        p = get_path(key)
        if p.exists():
            parts[key] = sha256_file(p)
    return hash_of_json(parts)
