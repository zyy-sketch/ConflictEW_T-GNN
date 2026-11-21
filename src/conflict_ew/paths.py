"""路径与配置管理：统一加载 config.yaml，解析项目内所有路径。

所有模块不得硬编码路径，一律经 :func:`get_path` 解析，
保证配置驱动（PRD §8.4 可维护性：超参不落代码，全在 config.yaml）。
"""
from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # 项目根 = 仓库根目录
CONFIG_FILE = PROJECT_ROOT / "config.yaml"


@lru_cache(maxsize=1)
def load_config() -> dict[str, Any]:
    """加载并缓存 config.yaml。"""
    import yaml

    with open(CONFIG_FILE, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if cfg is None:
        raise ValueError(f"config.yaml 为空或解析失败: {CONFIG_FILE}")
    return cfg


def cfg() -> dict[str, Any]:
    return load_config()


def get_path(key: str) -> Path:
    """按 config.yaml -> paths.<key> 解析绝对路径。"""
    path = cfg()["paths"][key]
    p = Path(path)
    if not p.is_absolute():
        p = PROJECT_ROOT / p
    return p


def ensure_dirs() -> None:
    """确保 data / artifacts / reports / web 等目录存在。"""
    for key in ("data_dir", "artifact_dir", "report_dir", "web_dir"):
        get_path(key).mkdir(parents=True, exist_ok=True)


def set_cwd() -> None:
    """将进程工作目录固定到项目根，保证相对路径一致。"""
    os.chdir(PROJECT_ROOT)
