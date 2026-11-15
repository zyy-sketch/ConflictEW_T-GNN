"""一键跑通全管道（PRD §10.2 整体流程验收：python scripts/run_all.py 一键跑通）。

用法：
    python scripts/run_all.py                 # 使用缓存数据跑全链路
    python scripts/run_all.py --refresh       # 强制重新抓取 WDI
    python scripts/run_all.py --no-report     # 跳过报告生成
    python scripts/run_all.py --no-web        # 跳过前端数据导出
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conflict_ew.pipeline import run_all  # noqa: E402


def main() -> int:
    # Windows GBK 控制台兼容：输出含 Unicode 符号（✅/⚠），强制 UTF-8 防止 UnicodeEncodeError
    for stream in (sys.stdout, sys.stderr):
        if stream is not None and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(encoding="utf-8", errors="replace")
            except (AttributeError, ValueError):  # pragma: no cover - 非常规流对象
                pass
    parser = argparse.ArgumentParser(description="ConflictEW 全管道")
    parser.add_argument("--refresh", action="store_true", help="强制重新抓取 WDI 数据")
    parser.add_argument("--no-report", action="store_true", help="跳过报告生成")
    parser.add_argument("--no-web", action="store_true", help="跳过前端数据导出")
    args = parser.parse_args()

    print("=" * 64)
    print("ConflictEW 军费冲突早期预警系统 — 全管道")
    print("=" * 64)
    summary = run_all(force_refresh=args.refresh,
                      generate_report_after=not args.no_report,
                      export_web=not args.no_web)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    if summary.get("data_source") == "simulated":
        print("⚠ 数据源为模拟降级数据（simulated_fallback）：WDI 不可达时自动回退，"
              "结果不可用于论文复现。请检查网络后重跑，或在 config.yaml 关闭 simulated_fallback。")
    print("=" * 64)
    print("✅ 管道完成。产物见 data/ 与 artifacts/，前端见 web/。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
