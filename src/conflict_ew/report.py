"""F17/F18 报告生成与导出归档。

- 月度/季度/自定义周期风险评估报告：风险概率 + SHAP 归因 + 推理链 + 基准对比
- 自包含 HTML（内嵌 base64 PNG 矢量图，无外链，F18 验收）
- PDF 通过浏览器打印另存（HTML 含打印样式）
- 历史报告归档 reports/index.json（检索/下载）
"""
from __future__ import annotations

import base64
import io
import json
from datetime import datetime
from typing import Any, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import meta
from .paths import cfg, get_path

# matplotlib 中文字体（Windows 微软雅黑，回退系统默认）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

RISK_COLORS = {"high": "#d62728", "mid": "#ff7f0e", "low": "#2ca02c"}


def _fig_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _risk_color(p: float, t: dict) -> str:
    if p >= t["high"]:
        return RISK_COLORS["high"]
    if p >= t["mid"]:
        return RISK_COLORS["mid"]
    return RISK_COLORS["low"]


def _chart_top_countries(top: pd.DataFrame, t: dict) -> str:
    """Top-N 高风险国家柱状图。"""
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [_risk_color(p, t) for p in top["p_conflict"]]
    ax.barh(top["country_name"].tolist(), top["p_conflict"].tolist(), color=colors)
    ax.axvline(t["high"], color="#d62728", ls="--", lw=1, label=f"高风险阈值 {t['high']}")
    ax.axvline(t["mid"], color="#ff7f0e", ls=":", lw=1, label=f"中风险阈值 {t['mid']}")
    ax.set_xlim(0, 1)
    ax.set_xlabel("冲突风险概率 (OOS 2015-2024 最新年)")
    ax.set_title("Top 高风险国家风险概率")
    ax.legend(loc="lower right", fontsize=8)
    ax.invert_yaxis()
    return _fig_to_base64(fig)


def _chart_region_risk(oos_latest: pd.DataFrame, countries: pd.DataFrame) -> str:
    """区域平均风险柱状图。"""
    reg = countries[["iso3", "region"]].rename(columns={"region": "region_name"}).merge(
        oos_latest[["country", "p_conflict"]], left_on="iso3", right_on="country", how="inner")
    grp = reg.groupby("region_name")["p_conflict"].agg(["mean", "max", "count"]).sort_values("mean", ascending=True)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.barh(grp.index, grp["mean"], color="#4472C4", alpha=0.85, label="区域平均风险")
    ax.barh(grp.index, grp["max"], color="#ED7D31", alpha=0.45, label="区域内最高风险")
    ax.set_xlabel("风险概率")
    ax.set_title("全球区域风险概览（2015-2024 最新年）")
    ax.legend(loc="lower right", fontsize=8)
    return _fig_to_base64(fig)


def _chart_country_shap(country: str, year: int, shap_df: pd.DataFrame,
                        factor_registry: list[dict]) -> str:
    """单国 SHAP 归因水平条形图。"""
    rows = shap_df[(shap_df["country"] == country) & (shap_df["year"] == year)]
    if len(rows) == 0:
        return ""
    rows = rows.sort_values("shap_value")
    names = {f["id"]: f["name"] for f in factor_registry}
    fig, ax = plt.subplots(figsize=(8, 3.8))
    vals = rows["shap_value"].tolist()
    labels = [f"{f} {names.get(f, '')}" for f in rows["factor_id"]]
    colors = ["#d62728" if v >= 0 else "#2ca02c" for v in vals]
    ax.barh(labels, vals, color=colors)
    ax.axvline(0, color="#333", lw=0.8)
    ax.set_xlabel("SHAP 贡献（对 logit(p)）")
    ax.set_title(f"{country} {year} 年因子归因（正=推高风险，负=压低风险）")
    return _fig_to_base64(fig)


def _render_metrics_html(metrics: dict) -> str:
    auc = metrics.get("auc", {})
    brier = metrics.get("brier", {})
    lt = metrics.get("lead_time", {})
    rows = [
        ("样本外 AUC", f"{auc.get('auc', '—'):.4f}" if auc.get('auc') else "—",
         f"95% CI [{auc.get('ci_lower', '—'):.4f}, {auc.get('ci_upper', '—'):.4f}]" if auc.get('ci_lower') else "—"),
        ("Recall（召回率）", f"{metrics.get('recall', {}).get('value', '—'):.3f}", "冲突域漏报代价高，优先召回"),
        ("Precision（精确率）", f"{metrics.get('precision', {}).get('value', '—'):.3f}", "阈值 0.7"),
        ("Brier Score", f"{brier.get('brier', '—'):.4f}", f"Reliability {brier.get('reliability', '—'):.4f}"),
        ("预警提前量", f"{lt.get('mean_lead_months', '—')} 个月" if lt.get('mean_lead_months') is not None else "—",
         f"覆盖率 {lt.get('coverage_rate', '—'):.1%}" if lt.get('coverage_rate') is not None else "—"),
    ]
    html = "<table class='mt'><tr><th>指标</th><th>值</th><th>说明</th></tr>"
    for r in rows:
        html += f"<tr><td>{r[0]}</td><td>{r[1]}</td><td>{r[2]}</td></tr>"
    # 基准对比
    html += "</table><h3>基准对比与消融</h3><table class='mt'><tr><th>模型</th><th>测试期 AUC</th><th>差异(AUC 差)</th><th>FDR 显著</th></tr>"
    for b in metrics.get("baseline_comparison", []):
        pv = b.get("p_value")
        diff = b.get("diff")
        html += (f"<tr><td>{b.get('baseline')}</td><td>{'—' if diff is None else f'{diff:+.4f}'}</td>"
                 f"<td>{'—' if pv is None else f'{pv:.4f}'}</td>"
                 f"<td>{'是' if b.get('fdr_significant') else '否'}</td></tr>")
    html += "</table>"
    return html


def generate_report(report_type: str = "quarterly", countries: Optional[list[str]] = None,
                    start_year: Optional[int] = None, end_year: Optional[int] = None,
                    data: Optional[dict] = None, out_path=None) -> dict:
    """F17 主入口：生成自包含 HTML 报告并归档。

    :param report_type: monthly | quarterly | custom
    :param countries: iso3 列表（None=全部）
    :param start_year/end_year: 时间范围（None=最新年）
    :param data: 数据上下文（默认从 artifacts 加载）
    :return: 报告元信息（report_id, file_path, status...）
    """
    if report_type not in cfg()["report"]["supported_types"]:
        raise ValueError(f"不支持的报告类型: {report_type}，可选 {cfg()['report']['supported_types']}")

    ctx = data or _load_report_context()
    oos = ctx["oos_test"]
    shap_df = ctx["shap_df"]
    chains = ctx["chains"]
    metrics = ctx["metrics"]
    countries_df = ctx["countries"]
    factors = ctx["factors"]

    latest = oos.sort_values("year").groupby("country").tail(1)
    if countries:
        latest = latest[latest["country"].isin(countries)]
    if start_year and end_year:
        latest = latest[(latest["year"] >= start_year) & (latest["year"] <= end_year)]
    latest = latest.merge(countries_df[["iso3", "name_zh", "region"]],
                          left_on="country", right_on="iso3", how="left")
    latest["country_name"] = latest["name_zh"].fillna(latest["country"])
    top = latest.sort_values("p_conflict", ascending=False).head(10)

    t = cfg()["eval"]["thresholds"]
    img_top = _chart_top_countries(top, t)
    img_region = _chart_region_risk(latest, countries_df)

    # 单国深度分析（Top 3）
    deep_sections = ""
    for _, r in top.head(3).iterrows():
        c, yr = r["country"], int(r["year"])
        img_shap = _chart_country_shap(c, yr, shap_df, ctx["factor_registry"])
        chain = next((ch for ch in chains if ch["country"] == c and ch["year"] == yr), None)
        chain_html = ""
        if chain:
            steps = "".join(
                f"<li><b>{s['factor_name']}</b>（贡献 {s['shap_contribution']:+.3f}）：{s['statement']}</li>"
                for s in chain["steps"])
            chain_html = f"<h4>推理链（后端: {chain['backend']}）</h4><ol>{steps}</ol><p class='sum'>{chain['summary']}</p>"
        deep_sections += f"""
        <div class="country">
          <h3>{r['country_name']}（{c}）— {yr} 年 风险概率 {r['p_conflict']:.3f}
              <span class="lv lv-{r['risk_level'] if 'risk_level' in r else ('high' if r['p_conflict']>=t['high'] else 'mid')}">
              {'高风险' if r['p_conflict']>=t['high'] else ('中风险' if r['p_conflict']>=t['mid'] else '低风险')}</span></h3>
          <div class="imgs">{img_shap}</div>
          {chain_html}
        </div>"""

    period_label = {"monthly": "月度", "quarterly": "季度", "custom": "自定义"}[report_type]
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    report_id = f"R{datetime.now().strftime('%Y%m%d%H%M%S')}"
    html = f"""<!DOCTYPE html>
<html lang="zh-CN"><head><meta charset="utf-8">
<title>ConflictEW {period_label}风险评估报告</title>
<style>
 body{{font-family:"Microsoft YaHei",sans-serif;color:#222;margin:32px auto;max-width:900px;line-height:1.6}}
 h1{{color:#1a3a6b;border-bottom:3px solid #1a3a6b;padding-bottom:8px}}
 h2{{color:#1a3a6b;margin-top:36px}} h3{{color:#333}} h4{{color:#555}}
 .lv{{padding:2px 10px;border-radius:10px;color:#fff;font-size:13px}}
 .lv-high{{background:#d62728}} .lv-mid{{background:#ff7f0e}} .lv-low{{background:#2ca02c}}
 .imgs{{margin:10px 0}} .imgs img{{max-width:100%;border:1px solid #ddd}}
 .sum{{background:#f5f7fa;padding:10px 14px;border-left:4px solid #1a3a6b}}
 table.mt{{border-collapse:collapse;width:100%;margin:10px 0}}
 table.mt th,table.mt td{{border:1px solid #ccc;padding:6px 10px;font-size:14px;text-align:left}}
 table.mt th{{background:#eef2f8}}
 .foot{{margin-top:40px;font-size:12px;color:#777;border-top:1px solid #ccc;padding-top:10px}}
 .risk-table{{width:100%;border-collapse:collapse;margin:10px 0}}
 .risk-table td,.risk-table th{{border:1px solid #ccc;padding:6px 10px;font-size:14px}}
 @media print{{.no-print{{display:none}} body{{max-width:100%}}}}
</style></head><body>
<div class="no-print" style="text-align:right">
 <button onclick="window.print()">导出 PDF（打印）</button></div>
<h1>ConflictEW {period_label}风险评估报告</h1>
<p><b>生成时间</b>：{now}　<b>数据周期</b>：2015-2024（OOS）　<b>模型版本</b>：{metrics.get('model_version', '—')}</p>
<h2>一、执行摘要</h2>
<p>本期共评估 {len(latest)} 个国家，其中高风险（p≥{t['high']}）{int((latest['p_conflict']>=t['high']).sum())} 个，
中风险（{t['mid']}≤p&lt;{t['high']}）{int(((latest['p_conflict']>=t['mid'])&(latest['p_conflict']<t['high'])).sum())} 个。</p>
<img src="data:image/png;base64,{img_top}">
<img src="data:image/png;base64,{img_region}">
<table class="risk-table"><tr><th>排名</th><th>国家</th><th>风险概率</th><th>等级</th><th>区域</th></tr>
{''.join(f"<tr><td>{i+1}</td><td>{r['country_name']}</td><td>{r['p_conflict']:.3f}</td>"
         f"<td>{'高风险' if r['p_conflict']>=t['high'] else ('中风险' if r['p_conflict']>=t['mid'] else '低风险')}</td>"
         f"<td>{r.get('region','—')}</td></tr>" for i, r in top.iterrows())}
</table>
<h2>二、单国深度分析（Top 3）</h2>
{deep_sections}
<h2>三、模型回测与基准对比</h2>
{_render_metrics_html(metrics)}
<h2>四、方法与伦理声明</h2>
<p>本报告由 ConflictEW 管道生成：Kalman LLT 滤波提取军费异常信号 e* → 6 因子特征 → XGBoost
（blocked expanding-window CV，防泄漏）→ SHAP 归因（局部精度 &lt;1e-6）→ 规则模板推理链。
固定种子 {meta.SEED}，全链路可复现。</p>
<p>输出为概率与不确定性区间，非二元预警；不构成政治判断或军事行动建议，部署判断权归人。</p>
<div class="foot">ConflictEW v{cfg()['project']['version']}　报告 ID：{report_id}　·　数据来源标记：{ctx.get('data_source','—')}　·　本报告自包含（无外链）</div>
</body></html>"""

    out_dir = get_path("report_dir")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_path or (out_dir / f"ConflictEW_{report_type}_{now.replace(':', '-').replace(' ', '_')}.html")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(html)

    # 归档索引
    idx_path = out_dir / "index.json"
    history = []
    if idx_path.exists():
        with open(idx_path, "r", encoding="utf-8") as f:
            history = json.load(f)
    record = {"report_id": report_id, "type": report_type,
              "period": f"{start_year or cfg()['model']['cv']['test_start']}-{end_year or cfg()['model']['cv']['test_end']}",
              "countries": countries or "全部", "generated_at": now, "file_path": str(out_path),
              "status": "success", "size_bytes": out_path.stat().st_size, "self_contained": True}
    history.insert(0, record)
    with open(idx_path, "w", encoding="utf-8") as f:
        json.dump(history, f, ensure_ascii=False, indent=1)
    meta.log_stage("report_generated", report_id=report_id, type=report_type, path=str(out_path))
    return record


def _load_report_context() -> dict:
    """从 artifacts/data 加载报告所需上下文。"""
    from .paths import get_path

    ctx = {}
    oos = pd.read_csv(get_path("oos_file"))
    ctx["oos_test"] = oos[oos["fold"] == "test"]
    ctx["shap_df"] = pd.read_csv(get_path("shap_file"))
    ctx["factors"] = pd.read_csv(get_path("features_file"))
    ctx["countries"] = pd.read_csv(get_path("external_dir") / "countries.csv")
    with open(get_path("metrics_file"), "r", encoding="utf-8") as f:
        ctx["metrics"] = json.load(f)
    chain_path = get_path("artifact_dir") / "reasoning_chains.json"
    with open(chain_path, "r", encoding="utf-8") as f:
        ctx["chains"] = json.load(f).get("chains", [])
    with open(get_path("artifact_dir") / "factor_registry.json", "r", encoding="utf-8") as f:
        ctx["factor_registry"] = json.load(f)
    with open(get_path("data_dir") / "data_manifest.json", "r", encoding="utf-8") as f:
        mf = json.load(f)
    ctx["data_source"] = mf.get("entries", {}).get("raw_real", {}).get("source",
                                                                       mf.get("entries", {}).get("raw_simulated", {}).get("source", "—"))
    return ctx


def list_reports() -> list[dict]:
    idx_path = get_path("report_dir") / "index.json"
    if not idx_path.exists():
        return []
    with open(idx_path, "r", encoding="utf-8") as f:
        return json.load(f)
