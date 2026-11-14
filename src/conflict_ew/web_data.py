"""前端数据导出：将所有分析产物序列化为 web/data/app_data.js。

前端（P01-P07）通过 <script src="data/app_data.js"> 加载，避免 file:// 下
fetch 本地 JSON 的 CORS 限制，双击 HTML 即可完整演示。
同时生成 GeoJSON 名称匹配（合规世界地图着色）。
"""
from __future__ import annotations

import json
from typing import Any

import numpy as np
import pandas as pd

from . import meta
from .paths import cfg, get_path

# GeoJSON 名称修正表（world-geo.json 官方名称 → 国家表 name_en）
MAP_NAME_ALIASES = {
    "United States": "United States of America",
    "United States of America": "United States",
    "Bosnia and Herzegovina": "Bosnia and Herz.",
    "Central African Republic": "Central African Rep.",
    "Cote d'Ivoire": "Côte d'Ivoire",
    "Czechia": "Czechia",
    "Dominican Republic": "Dominican Rep.",
    "DR Congo": "Dem. Rep. Congo",
    "Congo": "Congo",
    "Eswatini": "eSwatini",
    "North Macedonia": "Macedonia",
    "Myanmar": "Myanmar",
    "South Korea": "Korea",
    "North Korea": "Dem. Rep. Korea",
    "South Sudan": "S. Sudan",
    "Syria": "Syria",
    "Taiwan, China": "Taiwan",
    "Timor-Leste": "Timor-Leste",
    "United Arab Emirates": "United Arab Emirates",
    "Yemen": "Yemen",
    "Russia": "Russia",
    "Vietnam": "Vietnam",
    "Tanzania": "Tanzania",
    "Venezuela": "Venezuela",
    "Bolivia": "Bolivia",
    "Brunei": "Brunei",
    "Burkina Faso": "Burkina Faso",
    "Cape Verde": "Cape Verde",
}


def _load_geojson_names() -> set[str]:
    geo_path = get_path("web_dir") / "vendor" / "world-geo.json"
    if not geo_path.exists():
        return set()
    with open(geo_path, "r", encoding="utf-8") as f:
        geo = json.load(f)
    return {f.get("properties", {}).get("name") for f in geo.get("features", [])}


def _map_name_for(name_en: str, geo_names: set[str]) -> str | None:
    if name_en in geo_names:
        return name_en
    if MAP_NAME_ALIASES.get(name_en) in geo_names:
        return MAP_NAME_ALIASES[name_en]
    if MAP_NAME_ALIASES.get(name_en) and MAP_NAME_ALIASES[name_en] in geo_names:
        return MAP_NAME_ALIASES[name_en]
    return None


def _risk_level(p: float) -> str:
    t = cfg()["eval"]["thresholds"]
    if p >= t["high"]:
        return "high"
    if p >= t["mid"]:
        return "mid"
    return "low"


def export_web_data(out_path=None) -> dict[str, Any]:
    """导出 app_data.js，返回导出摘要。"""
    from scipy.stats import norm

    out_path = out_path or get_path("web_data_file")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    t = cfg()["eval"]["thresholds"]

    # ---- 加载产物 ----
    countries = pd.read_csv(get_path("external_dir") / "countries.csv")
    oos = pd.read_csv(get_path("oos_file"))
    panel = pd.read_csv(get_path("panel_file"))
    signals = pd.read_csv(get_path("kalman_signals_file"))
    kalman_params = pd.read_csv(get_path("kalman_params_file"))
    shap_df = pd.read_csv(get_path("shap_file"))
    shap_imp = pd.read_csv(get_path("shap_importance_file"))
    metrics = json.loads(get_path("metrics_file").read_text(encoding="utf-8"))
    alerts = json.loads(get_path("alerts_file").read_text(encoding="utf-8"))
    manifest = json.loads(get_path("manifest_file").read_text(encoding="utf-8"))
    cleaning = json.loads((get_path("data_dir") / "cleaning_report.json").read_text(encoding="utf-8"))
    panel_audit = json.loads((get_path("data_dir") / "panel_audit.json").read_text(encoding="utf-8"))
    factor_registry = json.loads((get_path("artifact_dir") / "factor_registry.json").read_text(encoding="utf-8"))
    report_index = json.loads((get_path("report_dir") / "index.json").read_text(encoding="utf-8")) \
        if (get_path("report_dir") / "index.json").exists() else []

    geo_names = _load_geojson_names()
    country_meta = []
    for _, r in countries.iterrows():
        mn = _map_name_for(str(r["name_en"]), geo_names)
        country_meta.append({
            "iso3": str(r["iso3"]), "cow_code": None if pd.isna(r.get("cow_code")) else int(r["cow_code"]),
            "name_en": str(r["name_en"]), "name_zh": str(r["name_zh"]), "region": str(r["region"]),
            "map_name": mn,
        })

    # ---- 预测（test 期，含 CI 近似）----
    test = oos[oos["fold"] == "test"].copy()
    z = norm.ppf(0.975)
    test["lo"] = 1 / (1 + np.exp(-(np.log(np.clip(test["p_conflict"], 1e-6, 1 - 1e-6) /
                                          (1 - np.clip(test["p_conflict"], 1e-6, 1 - 1e-6))) - z * 0.25)))
    test["hi"] = 1 / (1 + np.exp(-(np.log(np.clip(test["p_conflict"], 1e-6, 1 - 1e-6) /
                                          (1 - np.clip(test["p_conflict"], 1e-6, 1 - 1e-6))) + z * 0.25)))
    test["level"] = test["p_conflict"].map(_risk_level)
    test = test.merge(countries[["iso3", "name_zh", "region"]], left_on="country", right_on="iso3", how="left")
    test["name_zh"] = test["name_zh"].fillna(test["country"])
    preds = test[["country", "name_zh", "region", "year", "p_conflict", "lo", "hi", "level"]].to_dict(orient="records")

    # ---- 预测历史（val 折 + test，国家详情页年份切换）----
    hist = oos[["country", "year", "p_conflict", "fold"]].to_dict(orient="records")

    # ---- 趋势（压缩：每国 years/mil_log/e/anom）----
    trends: dict[str, dict] = {}
    for iso3, grp in signals.groupby("iso3"):
        yr = grp["year"].tolist()
        trends[str(iso3)] = {
            "years": yr,
            "mil": [None if pd.isna(v) else round(float(v), 3)
                    for v in panel[panel["iso3"] == iso3].sort_values("year")["mil_expense_log"]],
            "e": [None if pd.isna(v) else round(float(v), 4) for v in grp["e_star"]],
            "anom": grp["is_anomaly"].astype(int).tolist(),
        }

    # ---- SHAP 矩阵（country × year × F1-F6）----
    shap_pivot = shap_df.pivot_table(index=["country", "year"], columns="factor_id",
                                     values="shap_value").reset_index()
    shap_map: dict[str, dict] = {}
    for _, r in shap_pivot.iterrows():
        d = {}
        for f in ("F1", "F2", "F3", "F4", "F5", "F6"):
            v = r.get(f)
            d[f] = None if pd.isna(v) else round(float(v), 5)
        shap_map.setdefault(str(r["country"]), {})[int(r["year"])] = d

    # ---- 面板统计 ----
    panel_stats = {
        "n_rows": int(panel_audit.get("n_rows", len(panel))),
        "n_countries": int(panel_audit.get("n_countries", panel["iso3"].nunique())),
        "positive_rate": round(float(panel_audit.get("positive_rate", panel["y_onset"].mean())), 4),
        "years": [int(panel["year"].min()), int(panel["year"].max())],
        "audit_issues": panel_audit.get("issues", []),
        "audit_passed": panel_audit.get("passed", True),
    }

    # ---- 数据源状态表（P02）----
    source_status = []
    src_order = ["raw_real", "raw_simulated", "panel", "features", "kalman_signals",
                 "oos_predictions", "shap_values", "metrics",
                 "countries", "cow_conflicts", "polity_vdem", "icow", "atop", "cow_adjacency"]
    entries = manifest.get("entries", {})
    for key in src_order:
        e = entries.get(key)
        if e is None:
            continue
        source_status.append({
            "name": key, "version": e.get("version", "—"), "sha256": (e.get("sha256") or "")[:16],
            "frozen_at": e.get("frozen_at", "—"), "path": e.get("path", "—"),
            "status": "✅" if e.get("sha256") else "⛔",
        })

    # ---- 面板预览（P02：前 100 行，含 6 因子与 y 标签）----
    preview_cols = [c for c in ["iso3", "year", "mil_expense_log", "e_star", "gdp_growth", "inflation",
                                "unemployment", "polity_score", "vdem_ldi", "icow_claim", "atop_change",
                                "F1_e_star", "F2_arms_race", "F3_gdp_growth_lag1", "F3_inflation_lag1",
                                "F3_unemployment_lag1", "F4_polity_change", "F4_vdem_ldi_change",
                                "F5_icow_claim_lag1", "F6_atop_change_lag1", "y_onset"]
                       if c in panel.columns]
    panel_preview = panel[preview_cols].head(100).copy()
    for c in panel_preview.columns:
        if c not in ("iso3", "year"):
            panel_preview[c] = pd.to_numeric(panel_preview[c], errors="coerce").round(4)
    panel_preview = panel_preview.where(pd.notna(panel_preview), None)

    # ---- run-log（P02 采集日志展示，最近 20 条）----
    run_log = meta.load_run_log()[-20:]

    data = {
        "meta": {
            "project": cfg()["project"]["name"],
            "version": cfg()["project"]["version"],
            "seed": meta.SEED,
            "generated_at": meta.utcnow_iso(),
            "model_version": metrics.get("model_version", "—"),
            "data_source": manifest.get("data_source", "—"),
            "thresholds": t,
            "test_period": f"{cfg()['model']['cv']['test_start']}-{cfg()['model']['cv']['test_end']}",
            "pipeline_hash": meta.pipeline_hash(),
        },
        "countries": country_meta,
        "predictions": preds,
        "predHistory": hist,
        "trends": trends,
        "shap": shap_map,
        "shapImportance": shap_imp.to_dict(orient="records"),
        "kalmanParams": [{**r, "sigma2_eps": round(float(r["sigma2_eps"]), 6),
                          "sigma2_eta": round(float(r["sigma2_eta"]), 6),
                          "sigma2_zeta": round(float(r["sigma2_zeta"]), 6)} for r in
                         kalman_params.to_dict(orient="records")],
        "metrics": metrics,
        "factorRegistry": factor_registry,
        "alerts": alerts.get("alerts", []),
        "alertHistory": json.loads(get_path("alert_history_file").read_text(encoding="utf-8"))
        if get_path("alert_history_file").exists() else [],
        "reports": report_index,
        "manifest": manifest,
        "sourceStatus": source_status,
        "cleaningReport": cleaning,
        "panelStats": panel_stats,
        "panelPreview": panel_preview.to_dict(orient="records"),
        "runLog": run_log,
        "countryMap": {c["iso3"]: {"name_zh": c["name_zh"], "name_en": c["name_en"],
                                   "region": c["region"], "map_name": c["map_name"]} for c in country_meta},
    }

    js = "/* ConflictEW 前端数据（由 pipeline 自动生成，勿手改） */\n"
    js += f"window.APP_DATA = {json.dumps(data, ensure_ascii=False, separators=(',', ':'))};\n"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(js)
    meta.log_stage("web_data_export", path=str(out_path), size_bytes=out_path.stat().st_size,
                   n_predictions=len(preds), n_trends=len(trends))
    return {"size_bytes": out_path.stat().st_size, "n_predictions": len(preds),
            "n_trends": len(trends), "map_unmatched": sum(1 for c in country_meta if c["map_name"] is None)}
