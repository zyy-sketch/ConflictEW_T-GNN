"""全管道编排：数据 → 信号 → 因子 → 模型 → 归因 → 评估 → 推理 → 预警 → 报告。

与 PRD 阶段一一对应（F01→F18），供 scripts/run_all.py 一键调用。
每阶段写 run-log（meta.log_stage），产物统一落 data/ 与 artifacts/。
"""
from __future__ import annotations

import json
from typing import Any

import pandas as pd

from . import meta
from .alerts import append_alert_history, scan_alerts
from .build_panel import build_panel
from .data import clean_and_standardize, write_cleaned
from .data_fetch import get_military_and_economic_data
from .external import active_external_dir, ingest_external, merge_external_to_wdi
from .factors import build_features, save_factor_registry
from .kalman import compute_kalman_signals
from .metrics import evaluate
from .paths import cfg, ensure_dirs, get_path, set_cwd
from .report import generate_report
from .shap_analysis import compute_shap
from .xgb import MODEL_VERSION, train_and_predict


def finalize_panel(panel: pd.DataFrame, signals: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    """将 e_star/F1-F6 并入 panel.csv（最终形态对齐 PRD §8.1 字段表）。"""
    out = panel.merge(signals[["iso3", "year", "e_star", "is_anomaly"]], on=["iso3", "year"], how="left")
    feat_cols = [c for c in features.columns if c.startswith("F") and c not in ("F2_degraded",)]
    out = out.merge(features[["iso3", "year"] + feat_cols], on=["iso3", "year"], how="left")
    out.to_csv(get_path("panel_file"), index=False)
    return out


def country_names() -> dict[str, str]:
    ext = ingest_external()
    return dict(zip(ext.countries["iso3"], ext.countries["name_zh"]))


def run_all(force_refresh: bool = False, generate_report_after: bool = True,
            export_web: bool = True) -> dict[str, Any]:
    """执行完整管道，返回各阶段摘要。"""
    set_cwd()
    ensure_dirs()
    meta.init_run_log()
    meta.set_seed()
    meta.log_stage("pipeline_start", version=cfg()["project"]["version"], force_refresh=force_refresh)

    # ---- ① 数据获取（F01）----
    wdi = get_military_and_economic_data(force_refresh=force_refresh)
    data_source = wdi["source"]

    # ---- ② 外部数据接入（F02）----
    # external_source=simulated：按 §13.4 Mock 需求降级生成（固定种子可复现）。
    # external_source=real：读取官方原件转换产物（data/external_real，见 PROVENANCE.md），
    #   缺文件直接报错——真实模式下禁止静默生成模拟标签（R-8/P0-7）。
    EXT_GEN_VERSION = "8"
    use_real_external = (cfg().get("data", {}) or {}).get("external_source") == "real"
    ext_dir = active_external_dir()
    if use_real_external:
        missing = [f for f in ("countries.csv", "cow_conflicts.csv")
                   if not (ext_dir / f).exists()]
        if missing:
            raise FileNotFoundError(
                f"external_source=real 但 {ext_dir} 缺少 {missing}；"
                f"先运行 scripts/fetch_external.py --convert 生成官方原件转换产物")
    else:
        gen_ver_path = ext_dir / ".gen_version"
        ext_missing = (not (ext_dir / "countries.csv").exists()) or \
                      (not (ext_dir / "cow_conflicts.csv").exists()) or \
                      (not gen_ver_path.exists()) or \
                      (gen_ver_path.read_text(encoding="utf-8").strip() != EXT_GEN_VERSION)
        if ext_missing:
            from .simulate import write_external_from_wdi, write_simulated_data

            if data_source == "wdi-live" or data_source == "cache":
                write_external_from_wdi(wdi["df"])
            else:
                write_simulated_data()
            gen_ver_path.write_text(EXT_GEN_VERSION, encoding="utf-8")
    ext = ingest_external()
    if not use_real_external and len(ext.conflicts) == 0:  # 冲突标签缺失则强制生成（仅模拟模式）
        from .simulate import write_external_from_wdi

        write_external_from_wdi(wdi["df"])
        (get_path("external_dir") / ".gen_version").write_text(EXT_GEN_VERSION, encoding="utf-8")
        ext = ingest_external()

    # ---- ③ 清洗与标准化（F03）----
    merged = merge_external_to_wdi(wdi["df"], ext)
    clean, clean_report = clean_and_standardize(merged, ext.countries)
    write_cleaned(clean, clean_report)

    # ---- ④ 面板构建（F04）----
    panel, panel_audit = build_panel(clean)
    if not panel_audit.passed:
        meta.log_stage("panel_audit_failed", issues=panel_audit.issues)

    # ---- ⑤ Kalman 信号提取（F05）----
    signals = compute_kalman_signals(panel)

    # ---- ⑥ 多因子特征构造（F06）----
    features = build_features(panel, signals, ext.adjacency,
                              out_path=get_path("features_file"))
    save_factor_registry()

    # ---- ⑦ 模型训练与 OOS 预测（F07）----
    train_res = train_and_predict(features)

    # ---- ⑧ SHAP 归因（F08）----
    shap_res = compute_shap(features, train_res["run"].model_final, train_res["oos_df"],
                            calibrator=train_res["run"].calibrator)

    # ---- ⑨ 回测评估（F09）----
    metrics = evaluate(features, train_res["oos_df"], panel, train_res["fold_results"])

    # ---- ⑩ Agent 推理链（§8.3 / §13.4 规则模板 Mock）----
    from .agent import generate_for_oos, save_chains

    chains = generate_for_oos(train_res["oos_df"][train_res["oos_df"]["fold"] == "test"],
                              shap_res["shap_df"], features)
    save_chains(chains)

    # ---- ⑪ 预警（F15）----
    alerts = scan_alerts(train_res["oos_df"][train_res["oos_df"]["fold"] == "test"],
                         country_names=country_names(), chains=chains)
    append_alert_history(alerts)

    # ---- ⑫ 面板最终形态（含 e*/F1-F6）----
    finalize_panel(panel, signals, features)

    # ---- ⑬ 报告（F17/F18）----
    report_record = None
    if generate_report_after:
        report_record = generate_report(report_type=cfg()["report"]["default_type"],
                                        data={"oos_test": train_res["oos_df"][train_res["oos_df"]["fold"] == "test"],
                                              "shap_df": shap_res["shap_df"], "chains": chains,
                                              "metrics": metrics,
                                              "countries": ext.countries,
                                              "factors": features,
                                              "factor_registry": json.loads(
                                                  (get_path("artifact_dir") / "factor_registry.json").read_text(encoding="utf-8")),
                                              "data_source": data_source})

    # ---- ⑭ manifest（F03 数据清单）----
    meta.write_manifest(extra=[{"source": "wdi", "version": data_source, "path": str(get_path("raw_real_file")),
                                "sha256": meta.sha256_file(get_path("raw_real_file")),
                                "size_bytes": get_path("raw_real_file").stat().st_size,
                                "frozen_at": meta.utcnow_iso()}],
                        data_source=data_source)

    # ---- ⑮ 前端数据导出 ----
    if export_web:
        from .web_data import export_web_data

        export_web_data()

    meta.log_stage("pipeline_done", data_source=data_source, test_auc=metrics["auc"]["auc"])
    return {
        "data_source": data_source,
        "panel": {"n_rows": panel_audit.n_rows, "n_countries": panel_audit.n_countries,
                  "positive_rate": panel_audit.positive_rate, "audit_passed": panel_audit.passed},
        "kalman": {"n_countries": len(pd.read_csv(get_path("kalman_params_file"))),
                   "n_anomaly": int(signals["is_anomaly"].sum())},
        "model": {"model_version": MODEL_VERSION, "test_auc": metrics["auc"]["auc"],
                  "recall": metrics["recall"]["value"], "brier": metrics["brier"]["brier"]},
        "shap_local_accuracy": shap_res["local_accuracy"],
        "alerts": {"n_high": sum(1 for a in alerts if a["risk_level"] == "high"),
                   "n_mid": sum(1 for a in alerts if a["risk_level"] == "mid")},
        "report": report_record,
    }
