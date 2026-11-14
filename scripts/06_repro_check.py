"""06_repro_check.py 可复现性校验（PRD §8.2 / 11.1 / 13.4）。

同种子两跑 hash 一致性：
  - 默认模式：对已生成产物计算确定性 pipeline_hash 两次，必须一致
  - --full 模式：重跑核心计算链（清洗→面板→Kalman→特征→模型→SHAP→评估，
    数据用缓存），比较两轮 hash——严格验证"同种子两跑一致"

用法：
    python scripts/06_repro_check.py [--full]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from conflict_ew import meta  # noqa: E402
from conflict_ew.paths import cfg, ensure_dirs, get_path, set_cwd  # noqa: E402


def compute_hash_round(force_core: bool) -> str:
    """一轮 hash 计算；force_core=True 时重跑核心计算链。"""
    if force_core:
        set_cwd()
        ensure_dirs()
        meta.init_run_log()
        meta.set_seed()
        from conflict_ew.data import clean_and_standardize, write_cleaned
        from conflict_ew.data_fetch import get_military_and_economic_data
        from conflict_ew.build_panel import build_panel
        from conflict_ew.kalman import compute_kalman_signals
        from conflict_ew.factors import build_features
        from conflict_ew.xgb import run_blocked_cv
        from conflict_ew.shap_analysis import compute_shap
        from conflict_ew.external import ingest_external, merge_external_to_wdi

        wdi = get_military_and_economic_data()
        ext = ingest_external()
        clean, rep = clean_and_standardize(merge_external_to_wdi(wdi["df"], ext), ext.countries)
        write_cleaned(clean, rep)
        panel, _ = build_panel(clean)
        signals = compute_kalman_signals(panel)
        features = build_features(panel, signals, ext.adjacency,
                                  out_path=get_path("features_file"))
        run = run_blocked_cv(features)
        compute_shap(features, run.model_final, None)
    return meta.pipeline_hash()


def main() -> int:
    parser = argparse.ArgumentParser(description="可复现性校验")
    parser.add_argument("--full", action="store_true", help="重跑核心计算链后比较两轮 hash")
    args = parser.parse_args()

    set_cwd()
    print(f"[06_repro_check] 种子 = {meta.SEED}")
    h1 = compute_hash_round(force_core=False)
    print(f"[06_repro_check] 第一轮 pipeline_hash = {h1}")
    if args.full:
        print("[06_repro_check] --full 模式：重跑核心计算链...")
        h1 = compute_hash_round(force_core=True)
        print(f"[06_repro_check] 第一轮（重算后）pipeline_hash = {h1}")
    h2 = compute_hash_round(force_core=False)
    print(f"[06_repro_check] 第二轮 pipeline_hash = {h2}")
    if h1 == h2:
        print("✅ 可复现性校验通过：同种子两跑 hash 一致。")
        return 0
    print("❌ 可复现性校验失败：两跑 hash 不一致！请检查随机源/依赖版本。")
    return 1


if __name__ == "__main__":
    sys.exit(main())
