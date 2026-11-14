# ConflictEW_T-GNN 代码—论文一致性审查与开源准备清单

审查基准：论文 `Multi_revised_v4_submission.docx`（2026-11-14 投稿版，md5 `d1b32b66…`）
审查对象：`C:/Users/yeezy/Desktop/ConflictEW_T-GNN`（git master @ e54f97d）
审查日期：2026-11-14

---

## 0. 总体判断

仓库内存在**两套并行实现**，这是所有问题的根源：

| | 根目录 v1（git 已跟踪） | `new/` v2（**未被 git 跟踪**） |
|---|---|---|
| 数据 | WDI 抓取，40 国，1990–2024，n_test=375/10 阳性 | 165 国，**1970–2024**，test **1650/47 阳性** ✅ |
| 特征 | 12 列（含 log_level/growth/volatility/slope，无 V-Dem） | **9 列 F1–F6，命名与论文完全一致** ✅ |
| 模型 | XGB 200 轮，无校准，model 存 .pkl | XGB **300 轮 + isotonic 校准**，model 存 JSON ✅ |
| 基线 | logistic + VAR(2)（**把冲突标签放进 VAR 系统**） | logistic(固定效应) + **VAR(1) 简化版**（自注释承认） |
| 指标产物 | AUC 0.591 等，与论文**全部对不上** | AUC 0.5379 [0.4660,0.6098]、Brier 0.1202、τ*=0.507、lead 40.36 月(22/47)、消融 0.536/0.491/0.471/0.443 —— **与论文 TABLE III/消融逐一吻合** ✅ |
| Agent 层 | **模拟 κ 标注器 + 伪 SHAP fallback**（诚信风险） | 规则模板三机制 grounding ✅ |
| 测试 | 6 个文件 | 26 项测试（含 L1–L5 防泄漏自检）✅ |

**结论**：论文数值只能由 `new/` 复现；开源建仓必须以 `new/` 为主干。根目录 v1 若保留原样公开，其中的模拟 κ 与伪 SHAP 会被审稿人/读者视为学术诚信问题。

**附带发现（与论文投稿线联动）**：`new/artifacts/xgb_model.json`（332,450 B）与论文 `runtime_benchmark.json` 引用却缺失的模型文件**字节数完全一致**——可直接入 `release_package/artifacts/` 并登记 MANIFEST，解决论文整改清单 R-06（最硬可复现性缺口）。本仓库整理后也可直接作为论文 Data Availability 指向的代码仓库。

---

## 1. P0 — 论文不一致 / 诚信问题（开源前必须解决）

### P0-1 双实现并存，论文实现未入库
- 位置：根目录 vs `new/`；`new/` 整个目录 git untracked，且根目录有 5 个已修改未提交文件
- 问题：git master 上的代码（v1）产不出论文任何数值；论文实现只在本地 `new/`
- 修复：以 `new/` 内容重建仓库根（src/tests/config/scripts/web），v1 归档到 `legacy/` 分支或删除；提交并打 tag（如 `v1.0.0-paper`）与论文版本绑定

### P0-2 模拟 κ 标注器与论文口径直接冲突
- 位置：v1 `src/conflict_ew/agent/quality.py` + `pipeline.stage_agent()` + 产物 `artifacts/agent_quality.json`
- 问题：代码内置"两名高一致性标注者"模拟器（设计上保证 κ≥0.7），并输出 `authorized` 部署门。论文修订后口径为 **"planned, not executed…no κ value is reported"**——公开一个能产出 κ=0.7+ 的模拟器，等于自证 κ 是编的
- 修复：整模块删除；保留（或新建）①κ 协议的**规格文件**（holdout 50=25+25、三维定义、κ≥0.7 门槛——与论文 §III-C 逐字一致）；②真实标注用的空白标注表 + 汇总脚本（读入人工标注 CSV → 算 κ），不含任何打分逻辑

### P0-3 伪 SHAP fallback 与假置信区间
- 位置：v1 `pipeline.stage_agent()`：SHAP 缺失时按 `e_star×0.2`、`territorial_dispute×0.25` 等硬编码系数**合成"看起来合理"的 SHAP**；单点概率 CI 一律 p±0.1
- 问题：论文声称每条推理链的 SHAP 归因是 TreeExplainer 真实输出且 Σφ=logit(p) 校验通过
- 修复：删除 fallback（SHAP 不可得就跳过该链并记录）；CI 用模型真实校准分布或标注 "illustrative"

### P0-4 基线实现与论文描述不符（TABLE III 基线数值不可复现）
- 位置：`new/src/conflict_ew/metrics.py::baseline_logistic_re / baseline_var`；v1 `models/var.py`
- 问题：
  - 论文 LogReg(RE) = King–Zeng 稀有事件校正 [39]；仓库实现 = **国家固定效应虚拟变量 + C=0.1 + class_weight**（注释自认"近似"）→ 产出 AUC≈0.30 量级、diff +0.239 p=0.0，与论文 0.550 / diff −0.0118 p=0.824 矛盾
  - 论文 VAR(2) = **pooled 二阶向量自回归**（[军费增速, GDP 增速] 联立方程、训练窗 OLS、logistic link 校准）；仓库实现 = **VAR(1) 单方程**（仅军费增速，注释自认"简化实现"）→ 与论文 0.591、8/47、42 alerts、Wilson CI 无法对应
  - v1 的 VAR 更糟：把 `label` 作为第三列放进 VAR 系统
- 修复：按论文 §IV-B(1)(2) 逐句重写两个基线（King–Zeng 先验校正 logit；pooled VAR(2) 两方程 OLS + logistic link），输出必须对齐论文 `table3_unified.json` 的全部基线字段（AUC+DeLong、recall、flagged、onsets、Wilson precision、parametric-bootstrap Brier、paired p）

### P0-5 README / 仪表板品牌与论文矛盾
- 位置：根 `README.md`（标题"T-GNN Command Platform"、"模型 T-GNN（AUC 0.83）"、模拟新闻、"AI 风险分析"）
- 问题：论文是 Kalman+XGBoost 的**负结果/方法学定位**（AUC 0.538，基线全面占优）；"AUC 0.83" 与论文数值冲突，会被视为夸大
- 修复：README 以论文实现为主体重写（英文，含 Badges、论文引用、负结果诚实声明）；`T-GNN` 名称与论文不符（论文无图神经网络），要么弃用要么在 README 说明历史沿革；仪表板明示"demo, rules-based reasoning, not LLM"

### P0-6 数据源口径与论文不一致
- 位置：`config.yaml`/`new/config.yaml`（`MS.MIL.XPND.CD` = World Bank WDI 军费代理；`snapshot_date: 2024-06-20`）
- 问题：论文 TABLE IV 写 **"SIPRI Military Expenditure Database (2025)"** 且 Data Availability 指向 sipri.org；WDI 军费是 SIPRI 口径的转发但版本与快照日期不同
- 修复：或接入 SIPRI 官方数据文件（SIPRI 有可下载 xlsx），或在论文与仓库统一改口"WDI (MS.MIL.XPND.CD, SIPRI 口径)"并同步 TABLE IV/数据声明——两处必须一致，禁止"论文写 A 仓库用 B"

### P0-7 外部数据集溯源缺失（疑似按面板形状生成）
- 位置：`new/data/external/{atop,icow,polity_vdem,cow_conflicts}.csv`——四个文件**行数完全相同（8,506=面板行数+表头）**，每国每年一行无缺口；仓库内无任何下载/解析脚本
- 问题：真实 ICOW 只覆盖主张涉入国、真实 ATOP 是同盟 Dyadic 数据、真实 Polity 有 −66/−77 缺失码——面板完整的"外部数据"更像按面板骨架填充的。论文声称使用 ICOW Territorial Claims (2001)、ATOP (2002)、V-Dem、Polity IV 原始数据
- 修复：为每个数据源补 `scripts/fetch_<source>.py`（原始 URL、版本、下载日期、SHA-256），保留原始文件 + 转换脚本（原始→panel 格式），manifest 记录来源链。若部分字段确系估算/填充，必须如实在 README/manifest 标注并与论文核对

### P0-8 国别匿名化口径冲突
- 位置：论文 Fig.2/3/5 国家匿名（C1–C5，"mapping is not published"）；仓库以真实 ISO3 发布面板、6 国 Kalman 图（ISR/NGA/NOR/PHL/TUR/VEN）与 web 数据
- 问题：论文说身份映射不公开但"可从公开事件记录还原"；仓库直接公开映射等价物。两者不矛盾但口径要统一，否则审稿人会问"你说不公开，GitHub 上是什么"
- 修复：三选一——①论文与 Data Availability 同步改为"代码仓库含 ISO3 明细"；②仓库发布前抹掉 per-country 图与明细（只留聚合）；③论文 Fig 题注改为"coded, mapping available in the code repository"。推荐 ①（与论文 R02 整改联动）

### P0-9 论文承诺随文发布的工件在仓库缺失
- 论文正文/Data Availability 声明：`runtime_benchmark.json`、`bench_inference.py`、oos_predictions、shap_values、reasoning_chains、run_log、MANIFEST+SHA-256、**isotonic 校准后 Brier 0.030**
- 仓库现状：无任何 runtime bench 脚本；`metrics.json` 只有原始 Brier 0.1202（校准后 0.030 无输出物）；`oos_predictions.csv` 4,291 行**混合验证折与测试期**，未区分 `fold=="test"` 的 1,650 行发布口径
- 修复：从论文 revision_v3 `release_package/` 移植 bench 脚本与 MANIFEST 体系；`metrics.json` 增加校准前后 Brier 双口径；oos 输出拆分 `oos_test.csv`（1650 行）/`oos_val.csv` 并在 README 说明

### P0-10 论文有、代码无的评估项
| 论文项 | 论文值 | 仓库现状 |
|---|---|---|
| partial AUC (FPR 0–0.1) + bootstrap CI | 0.010 [0.000,0.072] vs 0.110 [0.044,0.186] | 无实现 |
| Brier skill score vs base rate | −3.344 → −0.078 | 无实现 |
| 基线 AUC 的 DeLong 区间 | 三模型均为 DeLong | 仅主模型有 |
| VAR precision Wilson CI / Brier parametric bootstrap | [0.038,0.221] / [0.254,0.271] | 无实现 |
| 插补敏感性（删 90 行重评） | 0.554 / 0.533 | 无实现 |
| 超参估计窗口敏感性 | pooled r=0.926、flag 一致 98.9% | 无实现 |
| 可靠性图 8 bin（宽 0.125，每 bin≥5 观测） | Fig.4b | `brier_bins: 10` 等宽，无 min-5 规则 |
| lead time 锚定首警、Fβ=2 训练期分位阈值 | 40.4 月、t*=0.507 | ✅ 已实现（唯一亮点，保留） |

修复：在 `metrics.py` 逐一补齐（论文数值即回归测试断言的 golden values）。

---

## 2. P1 — SCI 级可复现性（审稿人第一轮就会查）

1. **依赖锁定**：论文声明 `Python 3.13 / xgboost 3.4.1 / scikit-learn 1.9.0 / numpy 2.5.2 / pandas 3.0.5`；仓库 `requirements.txt` 全是宽松 `>=`，根 `pyproject.toml` 更旧（numpy>=1.24 等）。→ 生成 `requirements.lock`（pip freeze）与论文五处一致；pyproject 与 requirements 统一
2. **模拟降级默认关闭**：`new/config.yaml → simulated_fallback: true` 会在 WDI 抓取失败时**静默用假数据跑完整链**，产物与真数据无法区分。→ 默认 `false` + 显式 `--allow-synthetic` 旗标；manifest/日志中标注数据来源 real/simulated（现在有此字段，确认降级路径一定写入）
3. **随机种子与确定性**：seed 20240101 ✓ 贯穿配置；补 `06_repro_check.py` 的说明（两次连跑 hash 比对）进 README；bootstrap/DMatrix 线程数固定（论文 runtime 口径声明"thread count fixed"——补环境变量如 `OMP_NUM_THREADS` 进 run_all）
4. **数据清单对齐**：`data_manifest.json` 有 SHA-256 ✅，但缺论文 MANIFEST 的版本字段（v4-unified-2026-11-02 口径）；统一版本号体系，与论文 release_package MANIFEST 互认
5. **Kalman 估计窗口披露**：论文声明超参按各国**全 1970–2024 序列**估计（并做了 0.926/98.9% 敏感性）；确认 `new/` 拟合窗口与之逐字一致，并把敏感性脚本补进仓库（见 P0-10）
6. **防泄漏测试纳入 CI**：`tests/test_lookahead.py`（L1–L5）质量很好——加 GitHub Actions：`pytest -q` + `06_repro_check` 冒烟 + Golden 数值断言（AUC 0.5379/0.5497/0.5909、Brier 0.1202、消融四值、lead 40.36）

## 3. P2 — 开源工程化（建仓当日完成）

1. **仓库结构**（目标形态）：
   ```
   README.md            # 英文，论文摘要+负结果定位+复现三行命令
   LICENSE              # 代码 MIT/Apache-2.0（二选一，数据另算）
   CITATION.cff         # 引用条目（作者/标题/年份/DOI）
   requirements.txt / requirements.lock
   pyproject.toml       # 与 requirements 同步
   config.yaml          # 唯一配置源（=论文 Reproducibility Statement 引用物）
   scripts/             # 00_fetch_sipri → 01_external → 02_panel → 03_features
                        # → 04_train → 05_eval → 06_figures → 07_agent → 08_report
                        # → bench_inference.py → 99_repro_check.py
   src/conflict_ew/     # = new/src 骨架（kalman/factors/xgb/shap/metrics/agent/…）
   data/external/raw/   # 原始下载件（大文件走 Zenodo，git 存脚本+hash）
   tests/               # 26 项 + golden 值断言
   notebooks/demo.ipynb # 最小使用示例（单一国家 e* 提取→评分→推理链）
   docs/                # protocol_kappa.md（计划未执行的标注协议）、data_sources.md
   ```
2. **LICENSE 与数据合规**：SIPRI（CC BY 4.0，需署名）、World Bank（CC BY 4.0）、COW（公开学术数据，需引用）、V-Dem（需引用+许可条款）、ICOW/ATOP（学术使用）——README 加 "Data licenses" 节逐项列明；**数据文件本身建议不进 git**，由脚本下载 + Zenodo 冻结快照（与论文 release_package 口径一致）
3. **README 必备节**：论文链接/引用、安装、三行复现命令、期望输出（golden 数值表）、数据来源与许可、仓库结构、与论文工件对应表、免责声明（research-only，非决策系统——论文 §VI-C 自我实现预言风险表述可复用）
4. **版本与发布**：tag `v1.0.0` + GitHub Release 挂 Zenodo DOI（论文 R-01 要的可解析归档正是这个）；Release notes 列 golden 数值
5. **仪表板定位**：`web/` 保留但 README 明示"visualization demo, not part of the scientific claims"；删除/隔离 `PRD.md`、`REALTIME_SYSTEM_SPEC.md`、`SCI_OPTIMIZATION_PLAN.md`（内部文档，含大量与论文最终口径不符的历史表述——公开会被对照挑错）
6. **清理**：`__pycache__`、`.mplcache`、`.pytest_cache`、`cdata.json/db_data.json`（合成演示数据，注明）、`Multi.docx`（**论文手稿绝不能在公开仓库里**）、`results_report.html`、`dashboard_rt.html` 等根目录散落物全部清理或归档

## 4. 建议执行顺序

| 批次 | 内容 | 依赖 |
|---|---|---|
| ① | P0-1（new/ 提为主干）、P0-2/3（删 κ 模拟器与伪 SHAP）、P0-5（README 去品牌化） | 无 |
| ② | P0-4（重写两基线）+ P0-10（补 7 项指标）→ golden 值回归 | ① |
| ③ | P0-6/7/8（数据源统一与溯源）——需你确认 SIPRI 直取 vs WDI 口径、匿名化方案①②③ | 你的两个决策 |
| ④ | P1 全部 + CI | ② |
| ⑤ | P2 建仓收尾 → 与论文 release_package 合流（xgb_model.json 入包、MANIFEST 版本统一、Data Availability 填真实 URL） | ③④ |

批次 ③ 的两个决策（数据口径、匿名化方案）与论文修订线（整改项 R-02/R-06/R-08、I-01）联动，建议与第 1 步 GitHub 建仓一起定。
