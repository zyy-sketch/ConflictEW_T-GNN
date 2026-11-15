# 论文—仓库一致性全面修改计划

**依据**：`Multi_revised_v4_submission.docx`（2026-11-14 投稿版）× `CODE_REVIEW_论文一致性与开源准备审查.md` × 仓库实际核验（`new/artifacts/metrics.json`、`new/src/conflict_ew/metrics.py` 等，2026-11-14 复核）。

**总判断**：论文全部主模型数值可由 `new/` 复现（AUC 0.5379、Brier 0.1202、消融四值、lead 40.36 月）；但**两个基线（LogReg-RE、VAR(2)）的全部 Table III 数值、校准后 Brier 0.030、以及 7 项二级指标在仓库中不可复现**——这是修改计划的核心工作量。另有一批"论文承诺随文发布"的工件缺失。

---

## 一、论文所述方法、实验设置与结果梳理

### 1.1 方法（§III 三层架构）
| 层 | 论文描述 | 关键参数 |
|---|---|---|
| 信号提取 | 逐国 Kalman LLT（局部线性趋势），log(MILEX) 当前美元、不平减；标准化一步预测残差 e*，\|e*\|≥2 记异常 | 超参 Θ={σ²_eps,σ²_eta,σ²_zeta} 逐国 MLE，**估计窗=该国全序列 1970–2024**（34–55 观测），diffuse 初始化 1e6；训练窗重估敏感性 r=0.926、flag 一致 98.9% |
| 多因子评分 | XGBoost，9 列特征覆盖 F1–F6；无标准化；class weight；SHAP TreeExplainer，Σφ=logit(p) 校验 | 300 轮、depth 3、lr 0.05、row/col subsample 0.8、min_child_weight 2、λ(L2)=1.0、scale_pos_weight 自动、seed 20240101；模型存 325 KB JSON |
| Agent 推理 | 规则模板后端（发布运行口径，1,650 条链）+ 可选 LLM 后端；三机制 grounding；p>0.7 校验清单；tier 0.7/0.5 | κ 协议为"计划未执行"，无 κ 值报告 |

### 1.2 实验设置（§III-B/§IV/§V）
- 数据：SIPRI MILEX + WDI（GDP 增速/通胀/失业）+ COW（标签/邻接）+ Polity/V-Dem + ICOW + ATOP；165 国，1970–2024。
- 窗口：训练 1990–2014（4,109 行、3.41% 阳性，六因子缺失行剔除）；4 个扩展验证折（各 660 验证行）；测试 2015–2024（**1,650 行、47 阳性、2.85%**）。
- 阈值规则：验证折上最大化 Fβ=2 的分数分位数 → 测试窗一次套用（KF+MF τ=0.507/83 分位；LogReg τ=0.557/75 分位；VAR τ=0.513/55 分位）。
- 校准：isotonic 仅拟合于 4 个验证折（2,640 行、95 阳性），三模型同法。
- CI：AUC 用 DeLong；precision/recall/lead 用国家-年 bootstrap-1000；VAR 两项用 Wilson / 参数 bootstrap。

### 1.3 结果（§IV–§V，= 回归测试 golden values）
| 指标 | KF+MF | LogReg(RE) | VAR(2) |
|---|---|---|---|
| AUC | 0.538 [0.466,0.610] | 0.550 [0.460,0.639] | 0.591 |
| paired diff / p | — | −0.012 [−0.108,0.092], p=0.824 | −0.051 [−0.162,0.067], p=0.358 |
| pAUC(0–0.1) | 0.010 [0.000,0.072] | 0.110 [0.044,0.186] | 0.112 [0.051,0.225] |
| precision@100 | 0.020 [0.000,0.040] | 0.080 [0.030,0.130] | 0.070 [0.020,0.130] |
| recall | 0.106 | 0.319 | — |
| 告警数 / 覆盖 | 229 / 22 of 47 | 442 / 44 of 47 | 42 / 8 of 47 |
| precision | 0.022 | 0.034 | 0.095 [Wilson 0.038,0.221] |
| Brier 原始→校准 | 0.120→0.030 | 0.258→0.028 | 0.262→0.028 [参数 bootstrap 0.254,0.271] |
| lead time | 40.4 月 (22/47) | 42.8 (44/47) | 52.5 (8/47) |

- 消融：A1 0.536、A2 0.491、A3 0.471、A4 0.443（全模型 0.538，A1 差 0.002）。
- Brier 分解：reliability 0.092、uncertainty 0.0277、resolution≈0.00002；skill −3.344→−0.078。
- Fig.4c 混淆矩阵（τ*=0.507）：TN 1378 / FP 225 / FN 42 / TP 5。
- Fig.4b 可靠性图：8 bin（宽 0.125，每 bin≥5 观测）。
- 其他：插补敏感性 0.554/0.533（删 90 行）；SHAP 全局 |φ̄| F3=0.876、F4=0.716；Fig.3b 单例 p=0.827；runtime 0.189 ms mean / 0.155 ms 单行（n=50）；环境 Python 3.13 / xgboost 3.4.1 / sklearn 1.9.0 / numpy 2.5.2 / pandas 3.0.5；工件版本 v4-unified-2026-11-02 + MANIFEST(SHA-256)。

---

## 二、逐项差异对照表（论文 vs 仓库实际）

| # | 论文条目 | 论文值 | 仓库实际 | 定性 | 审查项 |
|---|---|---|---|---|---|
| D1 | 实现主干 | 论文数值的生成代码应随文发布 | `new/` 整目录 **git untracked**；git master 是 v1（数值全对不上） | 结构性 | P0-1 |
| D2 | 主模型指标 | AUC 0.5379、Brier 0.1202、消融 0.536/0.491/0.471/0.443、lead 40.36 (22/47)、τ*=0.507、CV 折记录 | ✅ `new/artifacts/metrics.json` 逐一吻合 | **一致（唯一已闭环项）** | — |
| D3 | LogReg(RE) 基线 | King–Zeng 先验校正、**无固定效应**，AUC 0.550、recall 0.319、442 告警、diff −0.012 p=0.824 | `metrics.py::baseline_logistic_re` = 国家固定效应虚拟变量 + C=0.1 + class_weight（注释自认"近似"），AUC≈0.30 量级、diff +0.239 p=0.0 | **不可复现** | P0-4 |
| D4 | VAR(2) 基线 | pooled 二阶 VAR，[军费增速,GDP 增速] 联立方程、训练窗 OLS、logistic link；AUC 0.591、42 告警、8/47、52.5 月、1,633 可评行 | `metrics.py::baseline_var` = VAR(1) 单方程（注释自认"简化实现"）；论文 §V-B 明言简化版"被本规范取代"但发布代码仍是简化版 | **不可复现** | P0-4 |
| D5 | isotonic 校准 | Brier 0.120→0.030；两基线 →0.028；reliability 0.092→0.002；skill −3.344→−0.078 | 仅原始 Brier 0.1202；**无任何校准代码/产物** | 缺失 | P0-9/10 |
| D6 | 二级指标 | pAUC+bootstrap CI、precision@100、Brier skill score、基线 DeLong、Wilson CI、参数 bootstrap | 全部无实现 | 缺失 | P0-10 |
| D7 | 敏感性分析 | 超参窗口 r=0.926/98.9%；插补敏感性 0.554/0.533 | 全部无实现（脚本与数值均缺） | 缺失 | P0-10 |
| D8 | 可靠性图 | 8 bin、宽 0.125、每 bin≥5 | config `brier_bins: 10` 等宽、无 min-5 规则 | 参数不符 | P0-10 |
| D9 | 数据源口径 | TABLE IV 写 "SIPRI Military Expenditure Database (2025/2026)"，Data Availability 指 sipri.org | config 用 WDI `MS.MIL.XPND.CD`（SIPRI 口径转发），snapshot 2024-06-20 | 口径不一致 | P0-6 |
| D10 | 外部数据集溯源 | 声称使用 ICOW/ATOP/V-Dem/Polity 原始数据 | `new/data/external/*.csv` 四文件行数完全相同（8,506 行），无下载/解析脚本 | **溯源缺失，疑似按面板填充** | P0-7 |
| D11 | 国别匿名化 | Fig.2/3/5 匿名 C1–C5，"mapping is not published" | 仓库/仪表板直接用真实 ISO3（ISR/NGA/NOR/PHL/TUR/VEN） | 口径冲突 | P0-8 |
| D12 | 随文工件 | runtime_benchmark.json、bench_inference.py、oos_predictions（测试期口径）、shap_values、reasoning_chains、run_log、MANIFEST+SHA-256 | 前两项缺；`oos_predictions.csv` 4,291 行混验证折与测试期，未拆分；其余有但不构成 MANIFEST 体系 | 部分缺失 | P0-9 |
| D13 | 依赖环境 | Python 3.13 / xgb 3.4.1 / sklearn 1.9.0 / numpy 2.5.2 / pandas 3.0.5 | requirements 全宽松 `>=`，pyproject 更旧 | 不可锁定复现 | P1-1 |
| D14 | κ 协议 | "planned, not executed"，不报告 κ 值 | v1 有模拟 κ 标注器（能产 κ≥0.7）+ 伪 SHAP fallback | **诚信风险（若公开 v1）** | P0-2/3 |
| D15 | 品牌与定位 | Kalman+XGBoost 负结果定位（AUC 0.538） | 根 README "T-GNN AUC 0.83" + 模拟新闻 | 矛盾 | P0-5 |
| D16 | 确定性声明 | thread count fixed、seed 固定、两次连跑可验证 | seed ✓；线程数未固定、无复现自查脚本入口说明 | 部分 | P1-3 |
| D17 | BSTS | 明确"仅设计、未估计、无代码随文发布" | 仓库无 BSTS 代码 | **一致**（保持缺失即可） | — |

---

## 三、逐项修改步骤（可执行）

### R-1 【D1】以 `new/` 重建仓库主干
1. `git checkout -b legacy-v1` 提交当前根目录状态作归档分支（保留 κ 模拟器历史但不公开）。
2. 在 `main` 上：把 `new/src`、`new/tests`、`new/config.yaml`、`new/scripts`、`new/web`、`new/requirements.txt` 移至根目录对应位置；根 v1 的 `src/`、`tests/` 删除（历史在 legacy 分支）。
3. `new/artifacts/` 全量复制入 `artifacts/`（git-ignore 产物目录，另入 release 包）。
4. 全部提交，打 tag `v1.0.0-paper`，与论文版本 `v4-unified-2026-11-02` 在 MANIFEST 中互相登记。
- **验证**：`git log` 干净、`pytest -q` 26 项通过、重跑管道 golden 值不变。

### R-2 【D14】删除模拟 κ 标注器与伪 SHAP fallback
1. 确认 `new/src/conflict_ew/agent.py` 无 κ 模拟逻辑（v1 的 `agent/quality.py` 随 R-1 一并消失）；grep `simulated|kappa|fallback` 确认为零。
2. 新增 `docs/protocol_kappa.md`：逐字复刻论文 §III-C（holdout 50=25+25、三维定义、κ≥0.7 门槛、双标注者）。
3. 新增 `data/annotation_sheet.csv`（空白标注表：chain_id × 3 维 × 2 标注者）+ `scripts/summarize_kappa.py`（读人工标注 → 计算 κ，**不含任何打分逻辑**）。
- **验证**：`pytest -q` 含 `test_agent.py`；`grep -ri "simulat" src/` 无 κ 相关命中。

### R-3 【D3】按论文规范重写 LogReg(RE) 基线（King–Zeng）
1. 重写 `metrics.py::baseline_logistic_re`：特征 = 原始（未滤波）log 军费增速 + 与主模型相同的 F3–F6 经济政治指标；**去掉国家固定效应**。
2. 实现 King–Zeng 先验校正：训练窗先按全样本拟合 logit，再按 `τ = ln[(π/(1−π))·((1−p)/(p))]`（π=训练期阳性率）做校正 logit 截距平移（King & Zeng 2001 §4.1 prior correction；勿用加权校正，与论文"class weighting 同"表述核对后择一并与论文同步措辞）。
3. 类不平衡处理与主模型同口径；训练/评估窗、阈值分位规则（75 分位 → τ=0.557）与论文一致。
4. 输出对齐 `table3_unified.json` 全部字段：AUC+DeLong、recall、alerts=442、onsets 覆盖 44/47、precision、Brier 原始/校准、paired bootstrap diff/p。
- **验证**：AUC 落在 0.550 附近且 DeLong CI ≈ [0.460,0.639]；若跑出数值不同 → **以重跑结果为准更新论文 Table III**（进入 R-9 论文回写流程），不许调参凑数。

### R-4 【D4】按论文规范重写 VAR(2) 基线
1. 重写 `metrics.py::baseline_var`：pooled 二阶 VAR，因变量向量 [Δlog MILEX, Δlog GDP]，两阶滞后，训练窗多方程 OLS。
2. 用拟合方程对测试窗逐年一步预测军费增速，经训练窗拟合的 logistic link 映射为概率。
3. 处理面板边缘损失两阶滞后 → 可评行 = 1,633；阈值取验证折 55 分位 → τ=0.513。
4. precision 用 Wilson 区间、Brier 用参数 bootstrap（论文口径），告警数目标 42、覆盖 8/47。
- **验证**：AUC ≈ 0.591（由排序计算、应免于概率尺度退化）；不达标则同 R-3 的"重跑回写"规则。

### R-5 【D5/D6】补齐校准与二级指标（一次进 `metrics.py` + 新脚本）
1. `calibrate_isotonic.py`：4 验证折合并拟合 isotonic（2,640 行/95 阳性）→ 输出三模型校准前后 Brier、reliability/resolution/uncertainty、skill score（对基准率）。
2. `metrics.py` 增加：`partial_auc(fpr_max=0.1)` + bootstrap CI；`precision_at_k(k=100)`；`brier_skill_score`；基线 DeLong 区间；Wilson；参数 bootstrap Brier。
3. 可靠性图改 8 bin（宽 0.125）+ 每 bin≥5 观测合并规则，输出 Fig.4b 数据。
4. 可靠性图、混淆矩阵（τ*=0.507：1378/225/42/5）、paired bootstrap 差值及 FDR——全部固化为 `scripts/07b_table3.py` 产出 `table3_unified.json`。
- **验证（golden 断言）**：Brier 0.120→0.030、基线 0.028、skill −3.344/−0.078、pAUC 三值、precision@100 三值、混淆矩阵四格——写成 `tests/test_golden_values.py`。

### R-6 【D7】补两项敏感性分析脚本
1. `scripts/09_kalman_window_sensitivity.py`：逐国仅用 1990–2014 重估超参 → 冻结参数前向滤波 → pooled Pearson r、median per-country r、RMSD、sign agreement、flag agreement、F1 相关；目标 r=0.926、98.9%。
2. `scripts/10_imputation_sensitivity.py`：删 90 行 F3 插补行（含 4/47 阳性）→ 主模型与 LogReg-RE 重评 → 0.554/0.533；两值须落在全面板 DeLong CI 内。
- **验证**：数值对不上时先查实现口径（窗口边界、标准差分母），不得直接采信论文数；确实复不出 → 论文回写实际值。

### R-7 【D8/D16】配置与环境对齐
1. `new/config.yaml`：`brier_bins: 8`、bin 宽 0.125、`min_bin_count: 5`；确认 Kalman 拟合窗口 = 各国全序列（与论文 §III-A 逐字一致）。
2. `run_all.py` 顶部固定 `OMP_NUM_THREADS`/`MKL_NUM_THREADS` 等；补 `scripts/99_repro_check.py`（两次连跑，比对 `metrics.json`+`oos_test.csv` SHA-256），README 写明入口。
3. 生成 `requirements.lock`（pip freeze），五个关键版本对齐论文声明；`pyproject.toml` 与之同步。若本机版本 ≠ 论文版本：**在正确版本环境中重跑全链**，golden 值漂移则回写论文。

### R-8 【D9/D10/D11】数据口径与溯源（✅ 决策已定，2026-11-14）
- **决策 A（已确认）：不换数据源**——保持 WDI `MS.MIL.XPND.CD` + 经济指标抓取管道不动；**论文侧改口**：TABLE IV 与 Data Availability 由 "SIPRI Military Expenditure Database" 改为 "World Bank WDI series MS.MIL.XPND.CD (SIPRI definition, current US$), snapshot 2024-06-20"，Acknowledgements 中 SIPRI 致谢改为口径来源说明；执行 R-8 前先做一次抽核（WDI 缓存 vs SIPRI 官方 xlsx 逐值抽 20 国比对），把"WDI 即 SIPRI 口径转发"坐实，核验脚本入 `scripts/verify_wdi_sipri.py`。
- **决策 B（已确认，方案①）：论文改口指向代码仓库**——Fig.2/3/5 题注及正文 "identities anonymized … mapping is not published" 改为 "identities coded as C1–C5 for readability; the ISO3 mapping is provided in the accompanying code repository"，与论文整改 R-02 联动；仓库侧无需抹数据。
1. 每个数据源补 `scripts/fetch_<source>.py`（原始 URL、版本、下载日期、SHA-256）+ 原始→面板转换脚本；`data/external/raw/` 留原始件或 Zenodo 冻结。
2. **对四个外部 CSV 做真实性核查**：与 ICOW/ATOP/V-Dem/Polity 官方原始文件逐字段抽样比对；发现估算/填充字段 → 在 README/manifest 如实标注，并同步修正论文 TABLE IV 的数据来源描述。
3. manifest 统一版本号 `v4-unified-2026-11-02`，与论文 release_package MANIFEST 互认。

**✅ 执行记录（2026-11-15 完成，commit 2d274d6）**：
- `scripts/fetch_external.py`：Polity5 / COW Intra-State Wars v5.1 / COW Direct Contiguity v3.2 三源自动下载（重复执行 SHA-256 逐字节一致）；ATOP/ICOW/V-Dem 官方站禁脚本直连 → 手动指引；全部来源+SHA-256 登记 `data/external/provenance.json`。
- **真实性核查结论（P0-7 坐实）**：`data/external/*.csv` 系 `simulate.write_simulated_data()` 按面板骨架生成的**模拟数据**（官方 Polity5 中 CHN 2000–2010 polity2 恒 −7，仓库文件为 −10~+10 随机）。影响 F4/F5/F6 特征、F2 邻接输入与 **y_onset 标签**——论文全部 golden 值建立在"真实 WDI 军费 + 模拟外部协变量/标签"栈上，R-11 回写必须对齐（如实披露，或接真实标签重跑全部数值）。
- 官方原件 → `data/external_real/` 转换（不覆盖 golden 基准）：polity.csv 149 国 6,667 行、cow_conflicts.csv 211 起 onset（AFG 1978 ✓）、cow_adjacency.csv 437 dyad；未匹配 103 实体入 unmatched.log。
- WDI↔SIPRI 抽核超额完成（100 国-年，超计划 20 国）：中位差 0、86% <1%、65/100 全同。
- 全链条文档 `PROVENANCE.md`（§3 诚实披露 / §4 SHA-256 / §8 R-11 联动）。

### R-9 【D12】补齐随文工件
1. 移植 `bench_inference.py`（批量 1,650 行 n=50 + 单行 n=50×20，DMatrix 排除）→ 产出 `runtime_benchmark.json`；复用 `new/artifacts/xgb_model.json`（332,450 B = 论文 325 KB ✓，已在审查中确认字节数吻合）。
2. `oos_predictions.csv` 拆分：`oos_test.csv`（fold=="test"，1,650 行，发布口径）/ `oos_val.csv`；README 说明。
3. `metrics.json` 增加校准前后 Brier 双口径；`shap_values.csv`、`reasoning_chains.json`、`run_log.jsonl`（backend=rule-template）入包；生成 `MANIFEST.json`（size+SHA-256 全量）。
- **验证**：MANIFEST 逐文件 hash 校验通过；论文 Data Availability 列举的每一项在包中可找到同名文件。

### R-10 【D13/D15】README/依赖/清理（开源当日）
1. 英文 README 重写：论文引用、负结果诚实声明（AUC 0.538、基线占优）、三行复现命令、golden 输出表、数据许可节（SIPRI/WB CC BY 4.0、COW、V-Dem、ICOW/ATOP）、"research-only" 免责（复用 §VI-C）、T-GNN 名称沿革说明或弃用。
2. 删除/隔离内部文档（PRD.md、REALTIME_SYSTEM_SPEC.md、SCI_OPTIMIZATION_PLAN.md、`Multi.docx`、`results_report.html`、`dashboard_*.html`、`cdata.json/db_data.json`、缓存目录）；仪表板明示 "demo, rules-based reasoning, not LLM"。
3. LICENSE（MIT/Apache 二选一）+ CITATION.cff + GitHub Actions（pytest + repro_check 冒烟 + golden 断言）+ tag/Release + Zenodo DOI（接收后）。

### R-11 论文侧回写清单（代码改动后触发）
- 若 R-3/R-4/R-6/R-7 重跑值与论文不符 → 以重跑为准回写：Table III 全行、§IV 基线段、§V-B/C 差值与 p 值、Fig.4b/6 数据与题注、TABLE IV 数据源措辞（决策 A）、Fig 题注匿名化口径（决策 B）、§III-A 拟合窗口措辞。
- 若数值全部复现 → 论文仅改：Data Availability 的仓库 URL（接收后去匿名）、版本号登记、（决策 A/B 的措辞同步）。

---

## 四、论文提及但仓库缺失内容清单（汇总）

| 缺失项 | 类型 | 补全方式 | 对应任务 |
|---|---|---|---|
| King–Zeng 稀有事件校正 logit | 代码模块 | 重写 `baseline_logistic_re` | R-3 |
| pooled VAR(2) 联立方程基线 | 代码模块 | 重写 `baseline_var` | R-4 |
| isotonic 校准及校准后 Brier 0.030 | 实验+产物 | `calibrate_isotonic.py` | R-5 |
| pAUC / precision@100 / skill score / DeLong(基线) / Wilson / 参数 bootstrap | 指标实现 | `metrics.py` 扩展 | R-5 |
| 8 bin 可靠性图数据 | 图表数据 | 图脚本改造 | R-5 |
| 超参窗口敏感性（r=0.926/98.9%） | 实验 | `09_…sensitivity.py` | R-6 |
| 插补敏感性（0.554/0.533） | 实验 | `10_imputation…py` | R-6 |
| bench_inference.py + runtime_benchmark.json | 代码+产物 | 移植+重跑 | R-9 |
| oos 测试期拆分预测文件 | 产物 | 拆分重导出 | R-9 |
| MANIFEST.json（SHA-256 体系） | 产物 | 生成器脚本 | R-9 |
| 数据源 fetch/溯源脚本 ×4+ | 代码 | fetch 脚本族 | R-8 |
| κ 协议规格 + 空白标注表 + 汇总脚本 | 文档+代码 | protocol_kappa.md 等 | R-2 |
| requirements.lock / LICENSE / CITATION.cff / CI | 工程 | 开源日完成 | R-10 |
| ~~BSTS 估计~~ | 实验 | **不需补**——论文明确"仅设计未执行"，保持缺失 | — |

---

## 五、优先级排列、预期效果与验证方法

| 批次 | 任务 | 优先级 | 预期效果 | 验证方法 | 依赖 |
|---|---|---|---|---|---|
| ① | R-1 主干重建 + R-2 删 κ/伪SHAP + R-10(README 部分) | **P0（诚信与结构，先做）** | 消除双实现与模拟器诚信风险；论文代码首次入库 | pytest 26 项过；grep 无模拟器残留；`git tag v1.0.0-paper` | 无 |
| ② | R-3 + R-4 两基线重写 | **P0（Table III 可复现性，最硬缺口）** | Table III 全部基线字段可由仓库复现 | golden 断言：0.550/0.591 及 CI、442/42 告警、diff/p 两对；不符则走 R-11 回写 | ① |
| ③ | R-5 + R-6 校准与二级指标、敏感性 | **P0（论文声称的分析全落地）** | 论文 §IV/§V 每个数字有对应产物文件 | `tests/test_golden_values.py` 全绿（Brier 0.030、skill、pAUC、混淆矩阵、0.926/98.9%、0.554/0.533） | ② |
| ④ | R-8 数据口径与溯源 | **P0.5（✅ 决策 A/B 已定，2026-11-14）** | 论文—仓库数据口径一致；溯源链完整 | fetch 脚本重跑 SHA-256 一致；外部 CSV 与官方原件抽样比对报告；WDI↔SIPRI 抽核 20 国 | 无（可与③并行） |
| ⑤ | R-7 + R-9 环境锁定与工件齐备 | P1 | 依赖可锁定、随文工件全量在包 | `99_repro_check` hash 一致；MANIFEST 校验脚本通过 | ③ |
| ⑥ | R-10 收尾 + R-11 论文回写 | P2 | 开源建仓 + 论文与仓库零矛盾 | CI 绿；论文 Data Availability 逐项指到包内文件；投稿前终检 | ④⑤ |

**总验证基线（批次③完成后必须全绿）**：AUC 0.5379 / Brier 0.1202→0.030 / 消融 0.536·0.491·0.471·0.443 / lead 40.36 (22/47) / τ*=0.507 / 混淆矩阵 1378·225·42·5 / 基线 0.550·0.591 / 敏感性 0.926·98.9%·0.554·0.533。

**风险与止损规则**：任何重跑值与论文 golden 值不符时，禁止反向调参凑数；一律以可复现结果为准回写论文（R-11），并在 response letter 中说明。若两基线按论文规范重写后数值系统性偏离（例如 King–Zeng 实现差异），优先核对：①先验校正公式方向 ②阈值分位规则 ③特征列完全一致——三处均无误仍不符，才考虑论文数值来源有误并全面回写。
