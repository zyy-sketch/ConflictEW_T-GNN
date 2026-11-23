"""κ 标注样本抽取（协议见 docs/protocol_kappa.md）。

分层抽取 50 条测试窗推理链（金标签 25 阳/25 阴，seed 20240101），打乱顺序
防止从排列推断金标签。输出两份标注表（A/B 各 50 行）+ 标注者阅读包 +
金标签映射（仅汇总用）。汇总：python scripts/summarize_kappa.py --sheet <合并表>

用法：python scripts/18_draw_kappa_sample.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from conflict_ew.build_panel import load_base_panel  # noqa: E402
from conflict_ew.paths import get_path  # noqa: E402

OUT = ROOT / "artifacts" / "annotation"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20240101
N_PER_CLASS = 25

# ---- PART2 ----
chains_doc = json.load(open(get_path("artifact_dir") / "reasoning_chains.json",
                            encoding="utf-8"))
chains = chains_doc["chains"]
feats = pd.read_csv(get_path("features_file"))
gold = {(r["iso3"], int(r["year"])): int(r["y_onset"]) for _, r in feats.iterrows()}
panel = load_base_panel()

pos = [(i, c) for i, c in enumerate(chains) if gold.get((c["country"], int(c["year"]))) == 1]
neg = [(i, c) for i, c in enumerate(chains) if gold.get((c["country"], int(c["year"]))) == 0]
rng = np.random.default_rng(SEED)
pick_pos = rng.choice(len(pos), N_PER_CLASS, replace=False)
pick_neg = rng.choice(len(neg), N_PER_CLASS, replace=False)
draw = [(pos[k][0], pos[k][1], 1) for k in pick_pos]
draw += [(neg[k][0], neg[k][1], 0) for k in pick_neg]
rng.shuffle(draw)

# ---- PART3 ----
sample = []
header_path = ROOT / "docs" / "annotation_packet_header.md"
n_pos = sum(1 for *_, g in draw if g == 1)
packet = header_path.read_text(encoding="utf-8").splitlines()
packet += ["", f"本包生成: scripts/18_draw_kappa_sample.py (seed {SEED}) | "
           f"样本 50 链（金标签阳/阴 = {n_pos}/{len(draw) - n_pos}，顺序已打乱）", ""]

for n, (idx, c, g) in enumerate(draw, start=1):
    cid = f"CH{n:03d}"
    country, year = c["country"], int(c["year"])
    sample.append({"chain_id": cid, "country": country, "year": year})
    src = panel[(panel["iso3"] == country) & (panel["year"] == year)]
    src_row = "" if src.empty else "；".join(
        f"{k}={src.iloc[0][k]:.3g}" for k in
        ("mil_expense_log", "gdp_growth", "inflation", "unemployment", "polity_score")
        if pd.notna(src.iloc[0][k]))
    steps = [f"  - {s.get('factor_name', s.get('factor_id'))}: {s['statement']}"
             for s in c.get("steps", []) if s.get("is_substantial", True)]
    packet += [f"## {cid} — {country} {year}", "",
               f"风险等级: {c.get('risk_level_label', c.get('risk_level'))}"
               f" | 概率: {c.get('probability'):.3f}", ""]
    packet += steps or ["  -（无实质因子陈述）"]
    concl = c.get("conclusion") or c.get("summary") or c.get("recommendation") or ""
    if concl:
        packet += ["", f"结论: {concl}"]
    packet += ["", f"源数据行: {src_row or '（缺失）'}", "", "---", ""]

header = [ln for ln in header_path.read_text(encoding="utf-8").splitlines()]
(OUT / "chain_packet.md").write_text(
    "\n".join(header + packet), encoding="utf-8")

gold_map = {f"CH{n:03d}": {"country": c["country"], "year": int(c["year"]), "gold": g}
            for n, (_, c, g) in enumerate(draw, start=1)}
(OUT / "kappa_gold_map.json").write_text(json.dumps(gold_map, ensure_ascii=False, indent=2),
                                         encoding="utf-8")
n_pos_final = sum(g for *_, g in draw)
print(f"DONE: 抽取 {len(draw)} 链（阳/阴 = {n_pos_final}/{len(draw) - n_pos_final}），"
      f"sheet_A/B 各 {len(draw)} 行 + chain_packet.md + kappa_gold_map.json")
