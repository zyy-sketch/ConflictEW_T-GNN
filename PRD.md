# ConflictEW 三层冲突早期预警系统 — 程序落地计划书（PRD）

> **文档类型**：程序落地计划书（Product Requirements Document）  
> **版本**：v1.0 ｜ **日期**：2026-08-18 ｜ **状态**：评审中  
> **依据论文**：《Multi-Indicator Conflict Early Warning: A Kalman Filter Approach with SIPRI Military Expenditure Data》（Multi.docx）  
> **对应章节**：§1.2 核心研究问题、§2.3–2.5 方法框架与创新点、§3 方法论、§4 回测验证、§5 评估框架  
> **对齐仓库**：`src/conflict_ew/` 可复现管道 + `scripts/` 流水线 + `config.yaml` 配置

---

## 1. 背景与目标

### 1.1 核心研究问题（论文 §1.2 原文转写）

> 能否用卡尔曼滤波器从各国军费支出时间序列中提取"真实军事意图信号"（剔除通胀、常规预算惯性等噪声），并结合经济、政治多维指标，构建一个具备可解释推理链的冲突早期预警系统？

该问题可拆解为三个子问题，对应三层架构：

| 子问题                         | 论文出处 | 落地目标                            |
| --------------------------- | ---- | ------------------------------- |
| Q1：如何从高噪声军费序列中分离"潜在战略信号"？   | §3.1 | 卡尔曼局部线性趋势（LLT）滤波 → 标准化一步预测残差 e* |
| Q2：多维度因子如何客观加权、形成可解释风险评分？   | §3.2 | XGBoost 监督训练 + SHAP 归因（不主观赋权）   |
| Q3：概率值如何转化为可审计、可复盘的"预警推理链"？ | §3.3 | 受约束 LLM/规则 Agent 生成自然语言推理链      |

### 1.2 项目目标

1. **方法落地**：把论文三层架构实现为可复现的端到端 Python 管道（数据 → 信号 → 因子 → 训练 → 回测 → 图件 → 报告）。
2. **论文支撑**：产出论文 §4/§5 全部数值（AUC/Precision/Recall/Brier/预警提前量、基准对比、消融），供 Multi.docx 填充 [tbd]。
3. **可解释交付**：每一条预警都有"因子级归因 + 自然语言推理链"，可被人工分析师审计。
4. **严格回测**：1990–2014 训练、2015–2024 样本外，与 COW 真实冲突事件匹配，杜绝事后拟合。

### 1.3 成功标准（可量化）

| 维度   | 目标             | 度量                                                                      |
| ---- | -------------- | ----------------------------------------------------------------------- |
| 判别力  | 样本外 AUC 显著优于基准 | AUC + DeLong 95% CI；与 logistic(RE)/VAR 的 DeLong 检验 p < 0.05（BH-FDR 校正后） |
| 召回优先 | 冲突预警不缺席        | Recall ≥ 0.5（冲突域漏报代价高于误报，论文 §2.5）                                       |
| 校准   | 概率有意义          | Brier 分解：Reliability 接近 0，AUC 与 Brier 同步报告                              |
| 可解释  | 全链路可审计         | 每条预警输出 SHAP 分解（Σφ_j = logit(p)）+ Agent 推理链；推理质检 Cohen's κ ≥ 0.7         |
| 可复现  | 同种子复跑结果一致      | `06_repro_check.py` 两跑 hash 一致；产物含版本戳/种子/run-log                        |
| 提前量  | 预警先于冲突         | 平均预警提前量（月）> 0 并尽量长                                                      |

---

## 2. 系统总体架构

### 2.1 三层架构（论文 §2.3 / §3）

```
┌─────────────────────────────────────────────────────────────────────┐
│ 第三层 Agent 推理层（§3.3）                                           │
│   概率 + SHAP → 结构化提示词 → 自然语言推理链（可审计、可复盘）        │
├─────────────────────────────────────────────────────────────────────┤
│ 第二层 多因子评分层（§3.2）                                           │
│   6 因子（军费异常/军备竞赛/经济压力/政治不稳/领土争端/同盟关系）       │
│   → XGBoost 加权 BCE + L2 → 冲突概率 p → SHAP 因子归因                │
├─────────────────────────────────────────────────────────────────────┤
│ 第一层 信号提取层（§3.1）                                             │
│   名义军费 log 变换 → 局部线性趋势状态空间模型 → 卡尔曼滤波           │
│   → 标准化一步预测残差 e*（±2σ 标记异常）                             │
├─────────────────────────────────────────────────────────────────────┤
│ 数据底座：SIPRI 军费 | COW 冲突/邻接 | WB/IMF 经济 | Polity/V-Dem     │
│           ICOW 领土 | ATOP 同盟                                       │
└─────────────────────────────────────────────────────────────────────┘
```

### 2.2 架构 ↔ 现有代码映射（已实现部分）

| 架构层  | 论文       | 现有实现                                                                           | 状态                                                |
| ---- | -------- | ------------------------------------------------------------------------------ | ------------------------------------------------- |
| 数据底座 | §3.1–3.2 | `src/conflict_ew/data.py`、`data_fetch.py`（WDI 自动抓取）、`config.yaml`              | ✅ 经济/军费已接真实数据；⛔ COW 标签/政治/领土/同盟因子未接               |
| 第一层  | §3.1     | `src/conflict_ew/features/kalman.py`（LLT + MLE 网格 + e*）                        | ✅ 已实现（`kalman_params.csv` / `kalman_signals.csv`） |
| 第二层  | §3.2     | `src/conflict_ew/features/factors.py`（9 维特征）、`models/xgb.py`、`eval/metrics.py` | ✅ 已实现；特征集合待按 §3.2 扩至 6 因子                         |
| 第三层  | §3.3     | `scripts/08_realtime_demo.py`（规则模板推理）                                          | ⛔ 仅规则模板 demo，未接真实 LLM / 质检                        |
| 回测   | §4/§5    | `scripts/03/04`、`eval/metrics.py`（DeLong/bootstrap/Brier/消融/基准）                | ✅ 已实现框架；数值待真实标签驱动                                 |
| 展示   | —        | `dashboard_v3.html`（展示）、`07_build_report.py`（results_report.html）              | ✅ 仅展示，不参与计算（设计约定）                                 |

### 2.3 数据流（管道视图）

**① 接入真实 COW 标签；② 明确前瞻性约束与阈值策略**

---

## 3. 功能需求（FR）

### FR1 数据接入与标签构建

**描述**：接入论文 §3.2 全部数据源，构建"国家 × 年"面板与冲突标签。

| 因子      | 指标                           | 数据源                                      | 现有状态                          |
| ------- | ---------------------------- | ---------------------------------------- | ----------------------------- |
| F1 军费异常 | Kalman 残差 e*                 | SIPRI（经 World Bank WDI `MS.MIL.XPND.CD`） | ✅ 已接入                         |
| F2 军备竞赛 | 邻国军费增速差值（加权）                 | SIPRI + COW 邻接矩阵                         | ⚠️ 有 `neighbor_e` 近似，未接真实邻接矩阵 |
| F3 经济压力 | GDP 增速/通胀/失业率                | World Bank / IMF                         | ✅ 已接入（3 维）                    |
| F4 政治不稳 | Polity IV 指数变化、V-Dem LDI 年变化 | Center for Systemic Peace / V-Dem        | ⛔ 未接入（需外部静态数据集）               |
| F5 领土争端 | 活跃领土主张二值变量                   | ICOW Territorial Claims                  | ⛔ 未接入                         |
| F6 同盟关系 | 正式军事同盟义务变化                   | ATOP Alliance Treaty Data                | ⛔ 未接入                         |

**标签（y）**：COW 冲突启动（12 个月内 onset，符合冲突四标准：动用武力 / ≥25 战斗死亡 / ≥2 方含国家政府 / 政府或领土不相容，Sarkees & Wayman 口径）。

**验收**：

- `data_manifest.json` 记录每个数据源的版本戳 / 下载日期 / SHA256 / 覆盖范围；
- 面板行数 ≈ 160+ 国 × 1990–2024，正样本占比 ~2–3% 并有明确口径说明；
- `panel.csv` 每行 =（country, year, 6 因子原始值, y）。

### FR2 信号提取层（Kalman LLT）

**描述**：对每国军费序列拟合局部线性趋势状态空间模型（论文 §3.1）：

```
观测方程：  y_{i,t} = μ_{i,t} + ε_{i,t}          ε ~ N(0, σ²_ε)
状态方程：  μ_{i,t} = μ_{i,t−1} + β_{i,t−1} + η   η ~ N(0, σ²_η)
            β_{i,t} = β_{i,t−1} + ζ               ζ ~ N(0, σ²_ζ)
异常信号：  e*_{i,t} = (y_{i,t} − ŷ_{i,t|t−1}) / √F_{i,t}
```

- 超参 Θ = {σ²_ε, σ²_η, σ²_ζ} 用预测误差分解极大似然 + 网格搜索（`config.yaml → kalman.grid_*`）估计；
- 初始状态扩散先验（方差 10⁶）；
- |e*| ≥ 2 标记异常（正异常 = 潜在扩军备战信号）。

**验收**：

- `kalman_params.csv` 含每国 σ²ε/σ²η/σ²ζ/log-lik/n；`kalman_signals.csv` 含 e* 与异常标记；
- 单国序列滤波 O(N)，40 国全年 < 1s（性能目标）；
- 已有 `tests/test_kalman.py` 保持通过。

### FR3 多因子评分层（XGBoost + SHAP）

**描述**：

- 训练目标：加权二元交叉熵 + L2（论文 §3.2 公式）：  
  `L(θ) = −Σ[w_pos·c·log p + w_neg·(1−c)·log(1−p)] + λ‖θ‖²`
  - `w_pos/w_neg` 与类频率成反比（稀有事件校正）；λ 网格搜索。
- 时序校验：**blocked expanding-window CV**——1990–2014 切 5 窗（1990–1994、1990–1999、…、1990–2014），按国家分组保时序，最近验证折调参，最终固定超参用于 2015–2024 OOS。
- 可解释性：训练后算 SHAP，满足局部精度 `φ₀ + Σφⱼ = logit(p)`；|φ| < 0.05 标注"非实质驱动因子"。

**验收**：

- `xgb_params.json` 记录超参/稀有事件权重/fold 划分/种子；
- `oos_predictions.csv` 为 blocked CV 各折样本外概率；
- 输出每 (country, year) 的 6 因子 SHAP 值（写入 `shap_values.csv`）。

### FR4 Agent 推理层

**描述**（论文 §3.3）：

- 输入槽位（固定模板）：(a) 国家与年份；(b) 综合风险概率 ± 95% CI；(c) Top-3 因子 SHAP 值及 z 解释；(d) 5 年滚动历史基线；(e) 邻国风险画像；(f) 约束指令"不得超出给定数据推断；因子贡献 < 0.05 log-odds 须声明非实质驱动"。
- **防幻觉三机制**：① 喂给 LLM 的数值预先计算、不可变；② 输出强制因子级引用（F1–F6 + 观测期）；③ 高风险（p > 0.7）追加核查清单（排除采购周期 / ≥2 独立因子超阈值 / 与其他数据源不矛盾）。
- **质检**：留出 50 条国家-年样本（25 冲突 / 25 非冲突），双人独立评分（事实准确性 / 逻辑连贯性 / 可操作性），Cohen's κ ≥ 0.7 才放行部署。
- 实现弹性：Phase 1 用确定性规则模板（现有 `08_realtime_demo.py`）先行打通；Phase 2 接入真实 LLM（GPT-4 类 / Llama-3 70B 离线部署）并在 `config.yaml` 切换 `agent.backend: rule|llm`。

**验收**：

- 示例推理链与论文 §3.3 示例同构（含 F1–F6 引用与 SHAP 数值）；
- 输出含因子 ID 引用 + 观测期；p > 0.7 时自动附加核查清单；
- 质检报告（评分 + κ）落盘 `artifacts/agent_quality.json`。

### FR5 回测评估（含基准与消融）

**描述**（论文 §4/§5）：

- **协议**：训练/调参/阈值校准全部在 1990–2014；2015–2024 仅访问一次做最终评估（预注册口径）。
- **指标**：Precision、Recall、AUC（DeLong 95% CI）、Brier（Reliability/Resolution/Uncertainty 分解）、预警提前量（月）。
- **基准**：
  1. logistic(RE)——稀有事件校正逻辑回归，用**原始（未滤波）**&#x519B;费增速 + 同经济政治指标（隔离 Kalman 贡献）；
  2. VAR(p=2)——军费、GDP 增速、冲突指标的线性互依（隔离状态空间贡献）。
- **消融**（§5.3）：
  - A1 多因子 only（原始增速，无 Kalman）——隔离信号提取贡献；
  - A2 Kalman + logistic（无 XGBoost）——隔离非线性建模贡献；
  - A3 Kalman + XGBoost（无 SHAP/Agent）——隔离可解释性贡献；
  - A4 全系统（Kalman + XGBoost + SHAP + Agent）——基线参照。
- **统计严谨性**：AUC 两两对比 DeLong 检验 + bootstrap p（1000 次国家-年重抽样）+ Benjamini–Hochberg FDR 校正（α=0.05）；全部指标给 95% CI。

**验收**：

- `metrics.json` 输出论文表 1 全部 [tbd] 数值（KF+MF / LogReg(RE) / VAR 三列 × 5 指标）；
- 消融四档结果 + 显著性汇总；`results/` 存明细 CSV。

### FR6 因果影响评估（Phase 2，可选）

**描述**：对测试期每个冲突事件，用 Bayesian structural time-series（Brodersen et al. 2015）构造合成反事实（匹配无冲突、经济政治画像相似的对照国家-年），估计异常军费信号对冲突概率的增量因果贡献及其 95% CI。

**验收**：`artifacts/causal_impact.json` 每事件输出（信号日、反事实、效应、95% CI）。

### FR7 可复现性与产物归档

**描述**：

- 全链路固定种子（`config.yaml → reproducibility.seed`），random/numpy/xgboost 统一；
- 所有超参集中配置，代码零魔法数字；逐阶段写 JSONL run-log（git hash / Python 与库版本 / 种子 / 时间戳）；
- `artifacts/` git-ignore，投稿时归档 Zenodo 并写 Data Availability；
- 论文图表全部由 `05_generate_figures.py` 生成矢量图（PDF/SVG，300 DPI，色盲安全调色板），HTML 只展示不计算。

**验收**：`06_repro_check.py` 两跑 hash 一致；`data_manifest.json` + run-log 齐全。

### FR8 展示与实时预警（增强，可选）

**描述**：

- `dashboard_v3.html`：只读展示（约定：不承担计算）；
- `07_build_report.py` → `results_report.html`：自包含投稿级结果报告；
- `08_realtime_demo.py` + `dashboard_rt.html`：SSE 流式实时看板（虚拟时钟回放 + /score），与 REALTIME_SYSTEM_SPEC.md 对齐，复用同一 `src/conflict_ew` 管道。

**验收**：`python scripts/08_realtime_demo.py --port 8765` 一键启动；前端可实时订阅预警流与 SHAP。

### FR9 真实 COW 标签接入（专项设计）

**背景**：当前 y 标签来自 `db_data.json` 策展验证集（与真实军费年份仅 17 起重叠，基准率 ~1.3%），是论文数字最大的不确定性来源（§10 R1）。本需求以 COW 官方数据构建 y，替换策展标签。

**数据版本与标签口径**：
- 主源：COW 州际冲突/战争数据。论文 §2.1 四标准（动用武力 / ≥25 战斗死亡 / ≥2 方且至少一方为国家政府 / 政府或领土不相容）对应 Sarkees & Wayman《Resort to War》口径（COW War Data v4.0，覆盖 1816–2007）。
- 测试期覆盖：2015–2024 官方版本可能滞后，采用 COW 中期更新（如 MID 5.0 派生）并在 `config.yaml → data.cow_version` + manifest 中**明确登记版本与覆盖边界**。
- 标签定义：`y_{it} = 1` ⟺ 国家 i 在 t 年后 **12 个月内进入新冲突 onset**（仅首个冲突年记 1；后续持续年记 0，避免重复计数）。与论文 §4"冲突 onset within 12 months"一致。

**构建流程**（`external.py` + 新增 `00b_fetch_external.py`）：
1. **落库**：`data/external/cow_*.csv` 原始文件 + 版本戳 + SHA256；
2. **国家编码映射**：COW code ↔ ISO3（面板主键）建立映射表，关键国家人工核对；未匹配国家写 `unmatched_countries.log`；
3. **派生 onset 标签表**：事件表 → 聚合为 `(country, year, onset)`；
4. **与面板 join**：以 (country, year) 主键；孤儿行（面板有、COW 无）默认记 0 并记录明细；
5. **口径校验**：正样本率应落在 ~2–3%；与策展标签做重叠一致性审计，不一致处人工复核。

**前瞻性说明**：标签使用 t 之后 12 个月的信息（这是"预测未来 onset"的必要窗口），但**特征只允许使用 ≤ t 的信息**——二者组合构成因果时序，详见 FR10。

**验收**：
- `data/external/` 落库 + manifest 版本戳；`label_table.csv`（country, year, onset, source, version）；
- 编码映射核对报告 + 标签一致性审计报告（`artifacts/label_audit.json`）；
- 测试期标签版本边界在 `data_manifest.json` 中显式声明。

### FR10 前瞻性约束与阈值策略（专项设计）

**A. 前瞻性约束（防泄漏清单，全管道强制）**

| # | 约束 | 落实位置 |
|---|---|---|
| L1 | **特征时序**：所有特征 X_{it} 只用 ≤ t 信息——e\* 为截至 t 的滤波输出；经济指标用 t−1 年值；邻国因子用 t−1 增速差 | `features/factors.py` + 测试断言 |
| L2 | **标签窗口**：y 表示 (t, t+12] 内 onset；(X_t, y_t) 配对天然满足"过去预测未来" | `external.py` 标签构建 |
| L3 | **参数估计时点**：Kalman 每国 σ²ε/σ²η/σ²ζ 只在训练期内估计，测试期不重估 | `features/kalman.py` + `pipeline.py` |
| L4 | **blocked CV**：expanding window（1990–1994 … 1990–2014），训练截止 < 验证起点 < 测试 2015；按国家分组，同国序列不跨折 | `models/xgb.py` |
| L5 | **调参与校准**：超参、λ、scale_pos_weight、概率校准（Platt/isotonic）、阈值全部只在训练期/最近验证折完成；2015–2024 仅访问一次（预注册） | `03_train_model.py` / `04_eval_backtest.py` |
| L6 | **数据冻结**：快照日期后不追加/更新；manifest 记录冻结时间与版本 | `meta.py` |
| L7 | **统计量禁用**：任何全样本统计量（全局均值/方差/分位）不得参与特征构造，一律在训练折内计算 | `features/factors.py` |

**B. 阈值策略**

1. **阈值不在测试期挑选**：在训练期验证折上用 Youden's J 与 Fβ（β=2，召回优先，对齐论文 §2.5 Hegre et al. 建议）初选候选阈值。
2. **三级运营阈值**（决策规则，写入 `config.yaml → eval.thresholds`）：
   - `p ≥ 0.7`：**高风险预警** —— 触发 Agent 核查清单（FR4 机制③）+ 人工复核（论文 §3.3 示例 p=0.85 属此档）；
   - `0.5 ≤ p < 0.7`：**中风险** —— 持续监控 + 定期复核；
   - `p < 0.5`：**低风险** —— 常规监控。
3. **校准优先于阈值**：先做概率校准（Platt/isotonic，训练期拟合），再设阈值，避免 raw 概率漂移导致操作点失真。
4. **阈值-指标透明**：测试期输出完整 PR 曲线 + 多阈值指标表（0.3 / 0.5 / 0.7 / 0.9 + Youden/Fβ 最优阈值下的 Precision/Recall/提前量），供论文表 1 与读者自选操作点。
5. **提前量口径耦合**：预警提前量依赖"穿越阈值的时刻"，报告时固定与 Recall 同一阈值口径，避免混用。
6. **稀有事件下阈值注意**：King–Zeng 先验校正后概率整体偏低，阈值应基于**校正后分布**设定，并在论文报告校正前后阈值对照。

**验收**：
- `tests/test_lookahead.py` 泄漏自检通过（构造"未来信息污染"用例，断言管道必须报错/结果恶化，防止回归）；
- `threshold_sweep.csv`（多阈值指标表）+ PR 曲线矢量图；训练期校准/阈值选择图；
- `eval.thresholds` 配置落盘，阈值选择过程全程仅在训练期完成（run-log 可证）。

---

## 4. 数据需求（DR）

| ID  | 数据                    | 用途      | 粒度   | 覆盖           | 获取方式                                  | 状态 |
| --- | --------------------- | ------- | ---- | ------------ | ------------------------------------- | -- |
| DR1 | SIPRI 军费（现价美元）        | F1 信号提取 | 国家×年 | 160+ 国 1949– | World Bank WDI `MS.MIL.XPND.CD`（自动抓取） | ✅  |
| DR2 | GDP 增速/通胀/失业率         | F3 经济压力 | 国家×年 | 全球           | World Bank WDI（自动抓取）                  | ✅  |
| DR3 | COW 冲突事件（onset 标签）    | y 标签    | 冲突-年 | 1816–        | correlatesofwar.org（需人工落库；构建细则见 FR9）      | ⛔  |
| DR4 | COW 邻接矩阵              | F2 军备竞赛 | 国家对  | 全球           | correlatesofwar.org（contiguity）       | ⛔  |
| DR5 | Polity IV / V-Dem LDI | F4 政治不稳 | 国家×年 | 全球           | systemicpeace.org / v-dem.net         | ⛔  |
| DR6 | ICOW 领土主张             | F5 领土争端 | 主张-年 | 全球           | icow.org                              | ⛔  |
| DR7 | ATOP 同盟条约             | F6 同盟关系 | 条约-年 | 1815–        | atop.rice.edu                         | ⛔  |

> **落地建议**：DR3–DR7 为静态数据集，建议以 `data/external/` 存放 CSV + `config.yaml → data.external.*` 登记版本戳；一次性人工下载落库后由 `00_fetch_data.py` 或新增 `00b_fetch_external.py` 归一化。

---

## 5. 非功能需求（NFR）

| ID   | 类别  | 要求                                                           |
| ---- | --- | ------------------------------------------------------------ |
| NFR1 | 性能  | 全管道单机 < 30 min（40 国基线）；Kalman 单国滤波 O(N)；Agent 推理 LLM 调用异步化   |
| NFR2 | 可复现 | 固定种子、配置驱动、run-log、hash 自检；发布前 Zenodo 归档                      |
| NFR3 | 伦理  | 输出概率 + 显式不确定性，而非二元预警（论文 §2.5 自证预言风险）；不输出政治判断，聚焦方法学与数据科学      |
| NFR4 | 安全  | 数据文件本地缓存；LLM 后端可选离线（Llama-3）部署；无外部 API key 硬编码（走 config/env） |
| NFR5 | 可扩展 | 因子以注册表方式扩展（新增因子 → 配置 + 一个特征函数）；支持新数据源插件                      |
| NFR6 | 可维护 | 类型注解、docstring、pytest 覆盖核心模块；超参不落代码                          |

---

## 6. 模块设计

### 6.1 目录结构（目标态）

```
src/conflict_ew/
├── config.py          # 配置加载/校验（已有）
├── data.py            # load_raw / build_panel / 标签构建（已有，扩展 COW 标签）
├── data_fetch.py      # WDI 自动抓取（已有）
├── external.py        # 【新增】COW/Polity/ICOW/ATOP 外部数据集归一化
├── meta.py            # manifest / run-log（已有）
├── pipeline.py        # 阶段编排 seed_everything（已有）
├── features/
│   ├── kalman.py      # LLT 滤波 + MLE 网格（已有）
│   └── factors.py     # 6 因子构造（已有 9 维，扩展 F4–F6）
├── models/
│   ├── xgb.py         # XGBoost + 加权 BCE + blocked CV（已有）
│   ├── logistic_re.py # 【新增】稀有事件校正逻辑回归基准
│   └── var.py         # 【新增】VAR(p=2) 基准
├── eval/
│   └── metrics.py     # AUC/DeLong/Brier/bootstrap/消融/对比（已有）
├── agent/
│   ├── template.py    # 【新增】提示词模板 + 槽位填充
│   ├── grounding.py   # 【新增】防幻觉三机制 + 核查清单
│   └── quality.py     # 【新增】双人标注 κ 质检
└── viz/figures.py     # 矢量图（已有）
scripts/               # 00–08 流水线（已有，新增 00b_fetch_external.py）
tests/                 # test_kalman/metrics/pipeline（已有，补 factors/agent/lookahead 测试）
config.yaml            # 全部超参（已有，补 external/agent 节）
```

### 6.2 关键接口约定

| 接口                          | 输入             | 输出                                | 说明                         |
| --------------------------- | -------------- | --------------------------------- | -------------------------- |
| `build_panel(raw)`          | raw 字典         | `panel.csv`（country, year, 指标, y） | y 由 COW 标签对齐（12 月 onset 窗） |
| `fit_kalman(series, grid)`  | 单国 log 军费序列    | σ²ε/σ²η/σ²ζ + e* 序列               | MLE 网格，扩散先验                |
| `build_features(panel)`     | 面板             | `features.csv`（e* + 6 因子 + 衍生量）   | 特征注册表驱动                    |
| `train_xgb(X, y, cv_folds)` | 特征 + 标签        | 模型 + fold 划分 + OOS 概率             | blocked expanding window   |
| `evaluate(oos, y_true)`     | 预测 + 真值        | 5 指标 + 95% CI + 显著性               | DeLong/bootstrap/FDR       |
| `agent_reason(record)`      | 因子 + SHAP + 概率 | 推理链 JSON                          | 槽位填充 + grounding + 核查清单    |

---

## 7. 技术栈

| 层     | 选型                                            | 备注                                                               |
| ----- | --------------------------------------------- | ---------------------------------------------------------------- |
| 语言    | Python 3.11+                                  | 托管 venv：`C:\Users\yeezy\.workbuddy\binaries\python\envs\default` |
| 状态空间  | 自实现（numpy）或 statsmodels                       | 论文 §3.1 LLT；自实现便于 MLE 网格与 e* 溯源                                  |
| 模型    | xgboost / scikit-learn                        | 加权 BCE；logistic(RE) 用 statsmodels 或自实现 King-Zeng                 |
| 时序基准  | statsmodels VAR                               | VAR(p=2)                                                         |
| 可解释   | shap                                          | 局部精度校验 Σφⱼ = logit(p)                                            |
| 统计    | scipy（DeLong/bootstrap）、statsmodels（Brier 分解） | FDR 用 statsmodels.multipletests                                  |
| 因果    | causalimpact / bsts 类实现                       | Phase 2 可选                                                       |
| Agent | Phase1 规则模板；Phase2 OpenAI/本地 Llama-3          | `agent.backend` 切换                                               |
| 图件    | matplotlib + seaborn                          | PDF/SVG、300 DPI、RdYlBu 色盲安全                                      |
| 报告    | 07_build_report.py → 自包含 HTML                 | 内嵌 base64 矢量图                                                    |
| 测试    | pytest                                        | `python -m pytest -q`                                            |

---

## 8. 里程碑计划

| 阶段               | 目标         | 关键任务                                                                                    | 验收                                         |
| ---------------- | ---------- | --------------------------------------------------------------------------------------- | ------------------------------------------ |
| **M0 现状盘点**      | 基线跑通       | `run_all.py` 全绿；确认 metrics.json 现状数值                                                    | 管道可复现，测试通过                                 |
| **M1 数据完备**      | 补齐论文全部数据源  | 接入 COW 冲突标签（FR9：编码映射 + onset 表 + 一致性审计）+ 邻接矩阵；落库 Polity IV/V-Dem/ICOW/ATOP；`external.py` + `00b_fetch_external.py` | `panel.csv` 含真实标签与 6 因子原始量；`label_audit.json`；manifest 记录版本   |
| **M2 完整训练与回测**   | 论文表 1 数值落地 | 扩 factors 至 6 因子；实现 logistic(RE) + VAR 基准；跑 blocked CV + 消融 A1–A4；阈值扫描与 PR 曲线（FR10）+ 泄漏自检 | `metrics.json` 填满表 1；`threshold_sweep.csv`；`test_lookahead.py` 通过 |
| **M3 Agent 推理层** | 可审计推理链     | `agent/` 三模块；先规则后 LLM；50 条样本双人质检                                                        | 推理链含 F1–F6 引用；κ ≥ 0.7；`agent_quality.json` |
| **M4 论文数字冻结**    | 投稿版结果      | 全管道最终跑；图件矢量输出；`results_report.html`；06 复现自检                                             | 数字冻结 + 两跑 hash 一致；Zenodo 归档准备              |
| **M5 发布与增强**     | 开源 + 实时    | 预注册（OSF/asPredicted）；因果影响（FR6）；实时看板（FR8）；git tag v1.0-submission                        | README 可一键复现；Data Availability 可兑现         |

---

## 9. 验收标准（发布 Checklist）

- [ ] `python scripts/run_all.py` 一键跑通，输出全部 `artifacts/*`
- [ ] `python -m pytest -q` 全部通过（含新增 factors/agent/external 测试）
- [ ] `metrics.json` 含：AUC+DeLong CI、Brier 分解、Precision/Recall、提前量、基准对比 p 值（FDR 校正）、消融 A1–A4
- [ ] `shap_values.csv` 满足局部精度 Σφⱼ = logit(p)（容差 < 1e-6）
- [ ] Agent 输出样例与论文 §3.3 示例同构；p > 0.7 自动附加核查清单
- [ ] `06_repro_check.py` 同种子两跑 hash 一致
- [ ] 每张图矢量（PDF/SVG）且色盲安全；图注含口径说明
- [ ] `data_manifest.json` 记录全部数据源版本戳与 SHA256
- [ ] `label_table.csv` 基于真实 COW 数据（FR9），含编码映射核对 + 一致性审计报告；测试期标签版本边界已声明
- [ ] `tests/test_lookahead.py` 泄漏自检通过；超参/校准/阈值选择全程仅用训练期数据（FR10，run-log 可证）
- [ ] `threshold_sweep.csv` 多阈值指标表 + PR 曲线；运营阈值（0.5/0.7）与论文 §3.3 示例对齐
- [ ] 论文表 1 无 [tbd]；Data Availability 指向 Zenodo/仓库

---

## 10. 风险与应对

| #  | 风险                                                                          | 等级   | 应对                                                                                |
| -- | --------------------------------------------------------------------------- | ---- | --------------------------------------------------------------------------------- |
| R1 | **真实标签不足**：策展冲突标签仅 17 起与真实军费年份重叠（基准率 ~1.3%），样本外 AUC 现状 0.648 远低于仪表板硬编码 0.83 | 🔴 高 | 接入真实 COW onset 标签 + Polity/ICOW/ATOP 协变量以提升判别；若仍有限，如实降调结论（概率校准 + 不确定性声明优先于虚高 AUC） |
| R2 | 外部静态数据（Polity/ICOW/ATOP）获取与格式归一化成本高                                         | 🟠 中 | M1 一次性人工落库；`external.py` 统一 schema；失败则按论文注明"待接入"并保持经济压力因子替代                       |
| R3 | 稀有事件导致指标方差大、CI 宽                                                            | 🟠 中 | King-Zeng 校正 + bootstrap 1000 次 + Wilson CI；报告 CI 而非单点                            |
| R4 | Agent LLM 幻觉 / 数据偏见                                                         | 🟠 中 | 三机制 grounding；双人质检 κ≥0.7 门槛；区域化校准；默认规则后端兜底                                        |
| R5 | 自证预言等伦理风险（论文 §2.5）                                                          | 🟡 低 | 概率化输出 + 显式不确定性；部署判断权归人                                                            |
| R6 | 卡尔曼线性高斯假设局限（阈值/体制转换）                                                        | 🟡 低 | 论文 §6.1 已声明；未来工作粒子滤波扩展                                                            |
| R7 | 数据许可/发布边界                                                                   | 🟡 低 | 只归档自产管道与缓存清单；第三方数据在 Data Availability 指向官网                                        |
| R8 | **COW 标签覆盖与编码映射错误**：测试期 2015–2024 官方版本滞后；COW↔ISO3 映射错配会系统性污染 y | 🟠 中 | FR9 流程：版本登记 + 映射人工核对 + 一致性审计；测试期用 MID 中期数据并标注；不可得则降级策展标签并在论文注明（诚实边界） |
| R9 | **泄漏回归**：新特征/新代码无意引入未来信息（如全样本标准化、测试期调阈值）                                | 🟠 中 | FR10 L1–L7 清单 + `tests/test_lookahead.py` 污染用例自检；阈值/校准仅训练期完成 |

---

## 11. 附录

### 11.1 论文章节 ↔ PRD 条款映射

| 论文              | PRD                    |
| --------------- | ---------------------- |
| §1.2 核心研究问题     | §1.1                   |
| §3.1 信号提取层      | FR2                    |
| §3.2 多因子评分层     | FR3 + FR1              |
| §3.3 Agent 推理层  | FR4                    |
| §4 回测验证         | FR5 + FR6 + FR9            |
| §5 评估框架（预注册/表1） | FR5 + FR9 + FR10 + §9       |
| §5.3 消融 A1–A4   | FR5                    |
| §4/§5 泄漏控制      | FR10（L1–L7 + 阈值策略）    |
| §6.1 局限         | §10 R6/R5              |
| 创新点 1–4         | §1.2 + FR2/FR3/FR4/FR5 |

### 11.2 因子定义表（论文 §3.2 浓缩）

| ID | 因子   | 指标                          | 方向              | 数据源              |
| -- | ---- | --------------------------- | --------------- | ---------------- |
| F1 | 军费异常 | e*（标准化一步预测残差）               | 正异常→风险↑         | SIPRI            |
| F2 | 军备竞赛 | 邻国军费增速加权差值                  | 差值↑→风险↑         | SIPRI + COW 邻接   |
| F3 | 经济压力 | GDP 增速/通胀/失业率               | 增速↓/通胀↑/失业↑→风险↑ | World Bank / IMF |
| F4 | 政治不稳 | Polity IV 指数变化、V-Dem LDI 变化 | 大幅负变→风险↑        | CSP / V-Dem      |
| F5 | 领土争端 | 活跃领土主张二值                    | 1→风险↑           | ICOW             |
| F6 | 同盟关系 | 正式同盟义务变化                    | 弱化/解散→风险↑       | ATOP             |

### 11.3 指标口径

- **AUC**：冲突/和平国家-年判别能力；DeLong 95% CI。
- **Brier**：`Brier = Reliability − Resolution + Uncertainty`（bins=10，`eval.brier_bins`）。
- **预警提前量**：风险概率穿越校准阈值 → COW 记录冲突 onset 的平均月数。
- **稀有事件校正**：XGBoost `scale_pos_weight`（`model.rare_event.pos_weight`）；logistic 基准 King–Zeng 先验校正（`prior_tau=0.5`，投稿前按领域先验校准）。
- **多重比较**：两两基准对比 p 值经 Benjamini–Hochberg 校正（`eval.alpha_fdr=0.05`）。

### 11.4 运营阈值速查（FR10 B 浓缩）

| 档位 | 区间（校准后 p） | 动作 | 对应论文 |
|---|---|---|---|
| 高风险 | p ≥ 0.7 | 触发 Agent 核查清单（机制③）+ 人工复核 | §3.3（示例 p=0.85） |
| 中风险 | 0.5 ≤ p < 0.7 | 持续监控 + 定期复核 | — |
| 低风险 | p < 0.5 | 常规监控 | — |

> 阈值在训练期验证折确定（Youden's J / Fβ=2），测试期固定；报告 PR 曲线 + 多阈值表，提前量与 Recall 共用同一阈值口径（FR10 B4–B5）。
