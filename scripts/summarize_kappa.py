"""汇总人工标注结果，计算 Cohen's κ（维度级 + 汇总），对照 0.7 部署门槛。

只做标注聚合，不含任何打分/模拟逻辑——标注值必须来自人工填写。
用法：python scripts/summarize_kappa.py --sheet data/annotation_sheet.csv
表格式见 docs/protocol_kappa.md 与 data/annotation_sheet.csv 表头。
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

DIMS = ("factual_accuracy", "logical_coherence", "actionability")
GATE = 0.7
VALID = {"pass", "fail"}


def read_sheet(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8-sig") as f:
        rows = [r for r in csv.DictReader(f) if any((r.get(c) or "").strip() for c in DIMS)]
    if not rows:
        sys.exit("错误：标注表为空（表头行不计）。请先由人工填写 pass/fail。")
    seen: set[tuple[str, str]] = set()
    for i, r in enumerate(rows, 2):
        key = (r["chain_id"].strip(), r["annotator"].strip())
        if key in seen:
            sys.exit(f"错误：第 {i} 行 chain_id×annotator 重复: {key}")
        seen.add(key)
        for dim in DIMS:
            v = (r[dim] or "").strip().lower()
            if v not in VALID:
                sys.exit(f"错误：第 {i} 行 {dim} 值非法（需 pass/fail）: {r[dim]!r}")
        if len({(r["annotator"] or "").strip() for r in rows}) < 2:
            sys.exit("错误：独立标注者不足 2 人，无法计算 κ。")
    return rows


def kappa(rows: list[dict[str, str]], dim: str) -> float:
    """Cohen's κ，按 chain_id 对齐两名标注者的 pass/fail。"""
    by_chain: dict[str, dict[str, str]] = defaultdict(dict)
    for r in rows:
        by_chain[r["chain_id"].strip()][r["annotator"].strip()] = (r[dim] or "").strip().lower()
    complete = [v for v in by_chain.values() if len(v) == 2]
    if not complete:
        sys.exit(f"错误：维度 {dim} 没有两名标注者齐全的 chain。")
    a1 = [v[list(v)[0]] == "pass" for v in complete]
    a2 = [v[list(v)[1]] == "pass" for v in complete]
    n = len(a1)
    p1, p2 = sum(a1) / n, sum(a2) / n
    po = (sum(x and y for x, y in zip(a1, a2))
          + sum((not x) and (not y) for x, y in zip(a1, a2))) / n
    pe = p1 * p2 + (1 - p1) * (1 - p2)
    if pe == 1.0:
        sys.exit(f"错误：维度 {dim} 边缘分布退化（全同值），κ 无定义。")
    return (po - pe) / (1 - pe)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sheet", required=True, help="人工填写后的标注 CSV 路径")
    args = ap.parse_args()

    rows = read_sheet(Path(args.sheet))
    print(f"有效标注行：{len(rows)}（chain×annotator）")
    values = {}
    for dim in DIMS:
        k = kappa(rows, dim)
        values[dim] = k
        print(f"  {dim:20s} κ = {k:+.3f}   门槛{GATE}: {'✅ 达标' if k >= GATE else '❌ 未达标'}")
    pooled = sum(values.values()) / len(values)
    gate_ok = all(k >= GATE for k in values.values())
    print(f"pooled κ = {pooled:+.3f}")
    print("部署门槛（每维度 κ ≥ 0.7）：", "✅ 授权" if gate_ok else "❌ 未授权（prospective gate 未通过）")


if __name__ == "__main__":
    main()
