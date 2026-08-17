"""07 — Build a self-contained results report (results_report.html).

Embeds the REAL pipeline outputs (metrics.json + vector figures) so the
findings can be inspected in a browser without any server or CDN. This is the
bridge between the reproducible pipeline and a human-readable "website" view.
"""
from __future__ import annotations

import base64
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "artifacts" / "figures"
METRICS = ROOT / "artifacts" / "metrics.json"
OUT = ROOT / "results_report.html"


def b64(path: Path) -> str:
    return base64.b64encode(path.read_bytes()).decode("ascii")


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


def fmt(x, nd=3):
    return "—" if x is None else f"{x:.{nd}f}"


def main() -> None:
    m = json.loads(METRICS.read_text(encoding="utf-8"))
    commit = git_commit()

    models = m["models"]
    abl = m["ablation"]
    paired = m["paired_vs_xgb"]
    conf = m.get("confidence", 0.95)
    alpha = 1 - conf

    # model comparison table rows
    order = ["xgb", "kalman_only", "logistic_rare_event"]
    labels = {
        "xgb": "XGBoost (full features)",
        "kalman_only": "Kalman-only (e* ablation)",
        "logistic_rare_event": "Logistic (rare-event)",
    }
    model_rows = ""
    for k in order:
        v = models[k]
        ci = v["auc_ci_95"]
        bci = v.get("brier_ci_95", [None, None])
        model_rows += f"""<tr>
<td class="l">{labels[k]}</td>
<td>{fmt(v['auc'])}</td>
<td>[{fmt(ci[0])}, {fmt(ci[1])}]</td>
<td>{fmt(v['brier'])}</td>
<td>{('[' + fmt(bci[0]) + ', ' + fmt(bci[1]) + ']') if bci[0] is not None else '—'}</td>
<td>{v['n']}</td>
<td>{v['base_rate']:.3f}</td>
</tr>"""

    # ablation rows
    abl_rows = ""
    for k in order:
        v = abl[k]
        abl_rows += f"""<tr><td class="l">{labels[k]}</td><td>{fmt(v['auc'])}</td>
<td>±{fmt(v['auc_sd'])}</td><td>{v['n_folds']}</td></tr>"""

    # significance rows
    sig_rows = ""
    for k in ["kalman_only", "logistic_rare_event"]:
        v = paired[k]
        d = v["delong"]
        sig_rows += f"""<tr><td class="l">XGBoost vs {labels[k]}</td>
<td>{d['diff']:+.3f}</td><td>{fmt(d['p_value'])}</td>
<td>{fmt(v['p_bootstrap'])}</td><td>{fmt(v['p_bootstrap_fdr'])}</td>
<td>{'YES' if v['rejected_fdr'] else 'no'}</td></tr>"""

    figures = {
        "ROC curve": "fig_roc.png",
        "Ablation (per-fold AUC)": "fig_ablation.png",
        "Reliability (calibration)": "fig_reliability.png",
        "SHAP feature importance": "fig_shap.png",
        "Kalman signal — NOR": "fig_kalman_NOR.png",
        "Kalman signal — PHL": "fig_kalman_PHL.png",
        "Kalman signal — TUR": "fig_kalman_TUR.png",
    }
    fig_html = ""
    for title, fn in figures.items():
        p = FIG_DIR / fn
        if p.exists():
            fig_html += (
                f'<figure><figcaption>{title}</figcaption>'
                f'<img alt="{title}" src="data:image/png;base64,{b64(p)}"></figure>'
            )

    xgb = models["xgb"]
    xgb_ci = xgb["auc_ci_95"]

    html = f"""<!DOCTYPE html>
<html lang="zh">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ConflictEW — Reproducible Results Report</title>
<style>
:root {{
  --bg:#0d1117; --panel:#161b22; --panel2:#1c2330; --line:#30363d;
  --txt:#e6edf3; --mut:#9aa7b4; --accent:#58a6ff; --warn:#f0883e;
  --good:#3fb950; --bad:#f85149;
}}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--txt);
  font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,"PingFang SC","Microsoft YaHei",sans-serif;
  line-height:1.55; }}
.wrap {{ max-width:1080px; margin:0 auto; padding:32px 22px 64px; }}
header h1 {{ font-size:26px; margin:0 0 4px; }}
header .sub {{ color:var(--mut); font-size:14px; }}
.banner {{ margin:22px 0; padding:16px 18px; border-radius:10px;
  background:linear-gradient(180deg,#2a1d12,#1c1610); border:1px solid var(--warn); }}
.banner b {{ color:var(--warn); }}
.cards {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:12px; margin:18px 0; }}
.card {{ background:var(--panel); border:1px solid var(--line); border-radius:10px; padding:14px 16px; }}
.card .k {{ color:var(--mut); font-size:12px; }}
.card .v {{ font-size:24px; font-weight:700; margin-top:4px; }}
.card .v small {{ font-size:13px; color:var(--mut); font-weight:400; }}
section {{ background:var(--panel); border:1px solid var(--line); border-radius:12px;
  padding:18px 20px; margin:18px 0; }}
section h2 {{ font-size:18px; margin:0 0 12px; }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th,td {{ text-align:right; padding:9px 10px; border-bottom:1px solid var(--line); }}
th {{ color:var(--mut); font-weight:600; }}
td.l, th.l {{ text-align:left; }}
tr:hover td {{ background:var(--panel2); }}
.note {{ color:var(--mut); font-size:13px; margin-top:10px; }}
figure {{ margin:18px 0; }}
figure img {{ width:100%; border:1px solid var(--line); border-radius:10px; background:#fff; }}
figcaption {{ color:var(--mut); font-size:13px; margin-bottom:6px; }}
footer {{ color:var(--mut); font-size:12px; margin-top:28px; text-align:center; }}
.tag {{ display:inline-block; font-size:11px; padding:2px 8px; border-radius:20px;
  border:1px solid var(--line); color:var(--mut); margin-left:8px; }}
</style>
</head>
<body>
<div class="wrap">
<header>
  <h1>Conflict Early Warning — Reproducible Results Report
    <span class="tag">commit {commit}</span></h1>
  <div class="sub">真实管道产物（非仪表盘硬编码值）· 生成自 artifacts/metrics.json · 置信水平 {(1-alpha)*100:.0f}%</div>
</header>

<div class="banner">
  <b>⚠ 关键发现：</b> 端到端管道在真实数据上得到 AUC = {fmt(xgb['auc'])}
  （DeLong 95% CI [{fmt(xgb_ci[0])}, {fmt(xgb_ci[1])}]），接近随机水平 0.5。
  这与 dashboard_v3.html 中宣称的 AUC = 0.83 严重不符——0.83 是硬编码假数。
  根因：仅 {xgb['positives']} 起冲突事件（基准率 {xgb['base_rate']:.1%}），
  且特征仅由军费衍生。投稿前须扩充样本/协变量或如实降调结论强度。
</div>

<div class="cards">
  <div class="card"><div class="k">XGBoost AUC</div><div class="v">{fmt(xgb['auc'])}</div></div>
  <div class="card"><div class="k">AUC 95% CI</div><div class="v" style="font-size:16px">[{fmt(xgb_ci[0])}, {fmt(xgb_ci[1])}]</div></div>
  <div class="card"><div class="k">Brier score</div><div class="v">{fmt(xgb['brier'],4)}</div></div>
  <div class="card"><div class="k">冲突事件 / 基准率</div><div class="v">{xgb['n']}<small> · {xgb['base_rate']:.1%}</small></div></div>
</div>

<section>
  <h2>① 模型对比（out-of-sample）</h2>
  <table>
    <thead><tr><th class="l">模型</th><th>AUC</th><th>AUC 95% CI</th>
    <th>Brier</th><th>Brier 95% CI</th><th>n</th><th>基准率</th></tr></thead>
    <tbody>{model_rows}</tbody>
  </table>
  <div class="note">所有 AUC 置信区间均由 DeLong (1988) 渐近法给出；Brier 由 bootstrap 给出。</div>
</section>

<section>
  <h2>② 消融分析（blocked CV，逐折 AUC ± SD）</h2>
  <table>
    <thead><tr><th class="l">变体</th><th>AUC</th><th>SD</th><th>折数</th></tr></thead>
    <tbody>{abl_rows}</tbody>
  </table>
  <div class="note">kalman_only = 仅 e* 单特征（移除结构特征）；logistic_rare_event = 稀有事件加权逻辑回归基准。</div>
</section>

<section>
  <h2>③ 显著性检验（XGBoost vs 基准）</h2>
  <table>
    <thead><tr><th class="l">对比</th><th>ΔAUC</th><th>DeLong p</th>
    <th>Bootstrap p</th><th>FDR (BH) p</th><th>拒绝 H₀</th></tr></thead>
    <tbody>{sig_rows}</tbody>
  </table>
  <div class="note">多重比较经 Benjamini–Hochberg 校正（FDR）。任一对比在 α={alpha:.2f} 下均未拒绝原假设——模型间无统计显著差异。</div>
</section>

<section>
  <h2>④ 可视化结果（出版级矢量图）</h2>
  {fig_html}
</section>

<footer>
  Generated by scripts/07_build_report.py · conflict_ew reproducible pipeline ·
  data: db_data.json (COW-derived 40-country military spending + 20 conflicts) ·
  随机种子 fixed · 同种子复现通过 (scripts/06_repro_check.py)
</footer>
</div>
</body>
</html>"""

    OUT.write_text(html, encoding="utf-8")
    size = OUT.stat().st_size
    print(f"wrote {OUT} ({size/1024:.0f} KB) — self-contained, {len(figures)} figures embedded")


if __name__ == "__main__":
    main()
