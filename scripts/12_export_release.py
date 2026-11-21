# -*- coding: utf-8 -*-
"""发布工件导出：oos 拆分 + MANIFEST.json（size + SHA-256，论文 Data Availability 口径）。

- oos_predictions.csv（验证折+测试期混合）拆分为：
    artifacts/oos_test.csv  — fold=="test"，1,650 行（论文发布口径）
    artifacts/oos_val.csv   — 验证折（校准/阈值选择的审计底稿）
- MANIFEST.json：覆盖 artifacts/ 全部发布文件 + 核心数据文件的
  size + SHA-256；version 字段与论文 release 包版本对齐。

用法：python scripts/12_export_release.py [--version v2.0.0-realdata]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conflict_ew.paths import get_path  # noqa: E402

VERSION_DEFAULT = "v2.0.0-realdata"


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--version", default=VERSION_DEFAULT)
    args = ap.parse_args()

    art = ROOT / "artifacts"
    art.mkdir(exist_ok=True)

    # ---- oos 拆分 ----
    oos = pd.read_csv(get_path("oos_file"))
    test = oos[oos["fold"] == "test"]
    val = oos[oos["fold"] != "test"]
    test_path = art / "oos_test.csv"
    val_path = art / "oos_val.csv"
    test.to_csv(test_path, index=False)
    val.to_csv(val_path, index=False)
    print(f"oos_test.csv: {len(test)} rows (期望 1650) | oos_val.csv: {len(val)} rows")

    # ---- MANIFEST：发布文件清单（artifacts 全量(含 figures/) + 核心数据文件 + κ 材料与标注表）----
    files = sorted(p for p in art.rglob("*") if p.is_file())
    extra = [get_path(k) for k in ("features_file", "panel_file", "kalman_signals_file",
                                   "kalman_params_file") if get_path(k).exists()]
    # 论文 Data Availability 声明随包发布的 κ 协议与人工标注表
    kappa_protocol = ROOT / "docs" / "protocol_kappa.md"
    annotation_sheet = ROOT / "data" / "annotation_sheet.csv"
    for p in (kappa_protocol, annotation_sheet):
        if p.exists():
            extra.append(p)
    entries = {}
    for p in list(files) + extra:
        if p.name == "MANIFEST.json" or p.is_dir():
            continue
        entries[str(p.relative_to(ROOT)).replace("\\", "/")] = {
            "bytes": p.stat().st_size, "sha256": sha256_of(p)}
    manifest = {
        "version": args.version,
        "generated_at": date.today().isoformat(),
        "note": "对每个发布文件登记 size 与 SHA-256；用于核验发布工件与记录运行产物一致。",
        "n_files": len(entries),
        "files": entries,
    }
    (art / "MANIFEST.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2),
                                       encoding="utf-8")
    print(f"MANIFEST.json: {len(entries)} files")
    # 自校验
    m = json.loads((art / "MANIFEST.json").read_text(encoding="utf-8"))
    bad = [k for k, v in m["files"].items()
           if sha256_of(ROOT / k) != v["sha256"]]
    print("自校验:", "OK" if not bad else f"FAIL {bad}")


if __name__ == "__main__":
    main()
