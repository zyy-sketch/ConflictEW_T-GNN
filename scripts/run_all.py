"""一键跑通全管道（PRD §10.2 整体流程验收：python scripts/run_all.py 一键跑通）。

用法：
    python scripts/run_all.py                 # 使用缓存数据跑核心链路（F01-F18）
    python scripts/run_all.py --refresh       # 强制重新抓取 WDI
    python scripts/run_all.py --no-report     # 跳过报告生成
    python scripts/run_all.py --no-web        # 跳过前端数据导出
    python scripts/run_all.py --extended      # 核心链路后追加完整复现工具链：
                                              #   09 Kalman 估计窗敏感性 + M3 价格诊断
                                              #   10 校准/可靠性/二级指标
                                              #   11 pAUC-precision@100 bootstrap CI
                                              #   14 τ/报警预算敏感性扫描
                                              #   15 训练窗重估端到端主结果
                                              #   13 论文级图表（fig1-5）
                                              #   12 发布工件导出 + MANIFEST
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conflict_ew.pipeline import run_all  # noqa: E402

EXTENDED_CHAIN = (
    "09_window_sensitivity.py",       # M2 估计窗敏感性 + M3 价格诊断
    "10_calibration_reliability.py",  # 校准/可靠性/8-bin 分解
    "11_secondary_metrics.py",        # pAUC / precision@100 / N6 插补稳健性
    "14_threshold_sensitivity.py",    # τ/报警预算扫描
    "15_train_window_refit.py",       # 训练窗重估端到端
    "13_generate_figures.py",         # 论文图表 fig1-5
    "12_export_release.py",           # MANIFEST（最后跑，覆盖全部产物）
)


def run_extended() -> None:
    """顺序执行扩展工具链；单步失败不阻断后续步骤（失败在输出中显式标出）。"""
    root = Path(__file__).resolve().parents[1]
    env = {**os.environ, "PYTHONUTF8": "1"}
    for script in EXTENDED_CHAIN:
        print(f"--- extended: {script} ---", flush=True)
        r = subprocess.run([sys.executable, str(root / "scripts" / script)],
                           cwd=str(root), env=env)
        if r.returncode != 0:
            print(f"⚠ {script} 失败（exit {r.returncode}）——继续后续步骤，"
                  "请单独重跑该脚本排查", flush=True)


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
    parser.add_argument("--extended", action="store_true",
                        help="核心链路后追加完整复现工具链（09→10→11→14→15→13→12）")
    args = parser.parse_args()

    print("=" * 64)
    print("ConflictEW 军费冲突早期预警系统 — 全管道")
    print("=" * 64)
    summary = run_all(force_refresh=args.refresh,
                      generate_report_after=not args.no_report,
                      export_web=not args.no_web)
    print(json.dumps(summary, indent=2, ensure_ascii=False, default=str))
    if summary.get("data_source") == "simulated":
        print("⚠ 数据源为模拟数据（source=simulated）：非官方 WDI 原件，结果不可用于论文复现。"
              "请检查网络后重跑；如非有意开启，请将 config.yaml 的 data.wdi.simulated_fallback 改回 false。")
    if args.extended:
        print("=" * 64)
        print("核心链路完成，开始扩展复现工具链（09→10→11→14→15→13→12）")
        run_extended()
    print("=" * 64)
    print("✅ 管道完成。产物见 data/ 与 artifacts/，前端见 web/。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
