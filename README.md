# Conflict Early Warning — T-GNN Command Platform

基于时序图神经网络（T-GNN）的全球冲突风险智能预警平台。

> 本项目包含两个互补产物：
> 1. **可复现研究管道（Python）** —— SCI 投稿的正式方法实现（数据 → Kalman 信号 → XGBoost blocked CV → DeLong/bootstrap CI → 矢量图）。见下文「可复现研究管道」。
> 2. **交互式仪表板（`dashboard_v3.html`）** —— 只做结果展示与交互，不承担计算职责。

---

## 可复现研究管道（Python · SCI 投稿用）

对齐 `SCI_OPTIMIZATION_PLAN.md` 与论文《Multi-Indicator Conflict Early Warning: A Kalman Filter Approach with SIPRI Military Expenditure Data》的 Methods / Reproducibility 要求，把原型中的硬编码数值替换为**从原始数据到指标的端到端可执行管道**。

### 安装

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

### 运行

```bash
python scripts/run_all.py              # 一键跑完 load → features → train → eval → figures
# 或分阶段：
python scripts/01_load_data.py
python scripts/02_feature_engineering.py
python scripts/03_train_model.py
python scripts/04_eval_backtest.py
python scripts/05_generate_figures.py
python scripts/06_repro_check.py       # 可复现性自检（同种子连跑两次比对 hash）
```

### 测试

```bash
pytest -q
```

### 产物（`artifacts/`，git-ignored，投稿时归档到 Zenodo）

| 产物 | 内容 | 对应计划条款 |
|------|------|--------------|
| `data_manifest.json` | 数据版本戳 / SHA256 / 快照日期 / SIPRI·COW 版本 | A3 |
| `panel.csv` / `features.csv` | 国家-年面板与 6 因子特征（e* 等） | A1 |
| `kalman_params.csv` | 每国拟合的 σ²ε/σ²η/σ²ζ/log-lik | A3 |
| `kalman_signals.csv` | 标准化一步预测残差 e* 与异常标记 | §3.1 |
| `oos_predictions.csv` | blocked CV 各 fold 的样本外预测 | B3 |
| `xgb_params.json` | 模型超参 / 稀有事件权重 / fold 划分 / 种子 | A3 |
| `metrics.json` | AUC·DeLong CI / Brier·bootstrap CI / 消融 / 两两对比（DeLong + bootstrap p + BH-FDR） | B1/B2 |
| `figures/fig_*.{pdf,svg,png}` | ROC、校准曲线、Kalman 信号、消融、SHAP（矢量图） | C |

### 关键设计（可复现性）

- **全链路固定种子**：`config.yaml → reproducibility.seed`，Python `random`/`numpy`/`xgboost` 统一。
- **配置驱动**：所有超参（Kalman 网格、XGBoost、CV 折数、阈值、先验校正）集中 `config.yaml`，无魔法数字。
- **严格样本外**：blocked expanding-window CV（`test_start: 2015`），特征用 `t-1` 之前数据，无 look-ahead 泄漏。
- **稀有事件处理**：XGBoost 用 `scale_pos_weight`；logistic 基准用 King–Zeng 先验校正（`prior_correction`）。
- **多重比较校正**：基准两两对比的 p 值经 Benjamini–Hochberg FDR 校正。
- **逐阶段溯源日志**：每个阶段写 JSONL run-log（git commit / Python·库版本 / 种子 / 时间戳）。

---

## 快速启动

直接双击打开 `dashboard_v3.html` 即可，无需安装任何依赖。

> 文件自包含：所有数据、GeoJSON 地图、CSS 和 JS 已内联。首次加载需联网（CDN 拉取 ECharts 和 D3.js），之后可离线使用（浏览器缓存）。

支持的浏览器：Chrome / Edge / Firefox 最新版（推荐 Chrome）。

---

## 功能概览

### 态势总览
- **真实世界地图**（GeoJSON）：点击任意国家进入详情页
- **TOP 10 风险排名**：实时异常得分排序
- **预警时间线**：2018–2024 年度预警统计

### 深度分析
- **动态国家关系图**（D3 力导向图）：拖拽 / 悬停 / 点击交互
- **因子雷达图**：6 维度风险因子分解
- **军费支出时序**：实测 vs 预测对比

### 预警中心
- 28 条历史预警记录（点击展开推理链）
- 按级别（HIGH / ELEVATED / MODERATE）筛选

### 数据中心
- 40 国基础信息表
- 军费支出查询（带趋势图）
- 双边关系网络
- 历史冲突事件
- 模型回测（消融实验 + ROC + 混淆矩阵）

### 国家详情页
- **可视化图表**：因子雷达 + 军费趋势 + T-GNN 异常信号
- **AI 风险分析**：6 模块固定模板
  - Executive Summary
  - Key Risk Drivers
  - Regional Context
  - T-GNN Signal Interpretation
  - 6-Month Outlook
  - Alert History
- **相关新闻**：6 条模拟国际新闻（带来源 / 日期 / 情感标签）

---

## 技术架构

| 层 | 技术 |
|------|------|
| 可视化 | ECharts 5.5 + D3.js v7 |
| 地图 | GeoJSON 世界地图（217 个区域） |
| 数据 | 内联 JSON（192KB 压缩） |
| 样式 | 原生 CSS（Command Steel 设计系统） |
| 模型 | T-GNN（时序图神经网络，AUC 0.83） |

---

## 数据说明

- **来源**：SIPRI 军费数据库（1990–2024）
- **覆盖**：全球 40 个主要国家
- **因子**：军费异常、军备竞赛、经济压力、政治不稳、领土争端、同盟紧张

| 文件 | 说明 |
|------|------|
| `dashboard_v3.html` | 主仪表板（自包含，可直接运行） |
| `cdata.json` | 紧凑格式数据（用于重建） |
| `db_data.json` | 全格式数据（带完整字段名） |

---

## 设计说明

配色方案 **Command Steel**：
- 深钢色背景 `#0b0f15`，避免纯黑
- 琥珀色强调 `#c9a84c`，避免 AI 审美的青蓝霓虹
- 6 色渐变风险热力图
- 等宽字体展示数据（SF Mono / JetBrains Mono）
- 暗哑红 / 钢青 / 低沉绿 语义色

参考：[OSINT War Room](https://github.com/Hue-Jhan/OSINT-War-Room)、Impeccable 设计原则

---

## 注意事项

1. 首次打开需联网（CDN 加载 ECharts + D3.js），后续浏览器缓存可离线使用
2. 建议分辨率 ≥ 1366×768，最佳 1920×1080
3. 模拟新闻来自模板生成，非实时抓取
4. AI 分析基于规则引擎 + 数据驱动模板，非真实 LLM 输出

---

## 方法论增强（对齐论文 Multi.docx）

按《Multi-Indicator Conflict Early Warning: A Kalman Filter Approach with SIPRI Military Expenditure Data》对仪表板做了针对性增强，主要用途与接口不变：

| # | 改动 | 论文依据 |
|---|------|----------|
| 1 | 新增本地线性趋势卡尔曼滤波（`kalmanLLT` / `fitKalman` / `computeKalmanSignals`）：对 SIPRI 军费做 log 变换后分解为潜趋势（水平 μ、斜率 β）与噪声，输出标准化一步预测残差 e* = v/√F | §3.1 信号提取层 |
| 2 | 以 ±2σ 阈值标记异常军费信号；国家详情页新增「Kalman Filter Signal (e*)」图（含 ±2σ 参考线） | §3.1 异常评分 |
| 3 | AI 分析因子归因改为 SHAP 风格，满足局部精度 Σφ_j = logit(p)，\|φ\|<0.05 标注「not a substantive driver」 | §3.2 多因子评分层 / SHAP |
| 4 | 高置信（p>0.7）时追加核查清单（采购周期 / 独立因子数 / 交叉来源），并加入「不超出给定数据推断」grounding 约束 | §3.3 智能体推理层 |
| 5 | 数据中心「Backtest」新增 Brier 分数及 Reliability / Resolution / Uncertainty 分解，消融实验扩展为基准对比（含稀有事件校正逻辑回归 + VAR(p)），并校准 Precision/Recall/F1 使其与混淆矩阵一致 | §2.5 / §4 评估框架 |

> 超参数（σ²_η、σ²_ζ、σ²_ε）在浏览器内通过预测误差分解的极大似然网格搜索估计；滤波为线性 O(N)，40 国全年序列 < 50ms。

