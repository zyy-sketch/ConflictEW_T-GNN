# PROVENANCE — 数据溯源与来源链登记

> 登记生成：`scripts/fetch_external.py`（manifest：`data/external/provenance.json`）
> 最后更新：2026-11-15
> 面板口径：40 国 × 1970–2024，军费 = WDI `MS.MIL.XPND.CD`（决策 A，2026-11-14 固化）

---

## 1. 总览

| 数据 | 角色 | 来源 | 状态 | 溯源等级 |
|---|---|---|---|---|
| 军费 / GDP 增速 / 通胀 / 失业率 | 主特征（F1–F3） | World Bank WDI API（实时抓取） | ✅ 真实，自动 | URL + 抓取时间戳 + SIPRI 抽核 |
| **冲突标签 y（onset/active）** | **y 标签** | **UCDP Intrastate Country-Year Onset v25.1（1946–2024）** | ✅ 真实，自动 | URL + 版本 + SHA-256 |
| Polity 政体指数 | 协变量（F4） | Center for Systemic Peace, Polity5 | ✅ 真实，自动下载 | URL + 版本 + SHA-256 |
| 国家邻接 | 协变量（F2 邻国项） | COW Direct Contiguity v3.2 | ✅ 真实，自动下载 | URL + 版本 + SHA-256 |
| COW 内战国-年（历史对照） | 参照 | COW Intra-State Wars v5.1（覆盖止于 ~2015） | ✅ 真实，自动下载 | URL + 版本 + SHA-256 |
| 同盟义务变化 | 协变量（F6） | ATOP 5.1 | ⚠️ 待手动下载 → F6 降级为 0 | 指引见 §6 |
| 领土主张 | 协变量（F5） | ICOW Territorial Claims | ⚠️ 待手动下载 → F5 降级为 0 | 指引见 §6 |
| V-Dem LDI | 协变量（F4 之 vdem 分量） | V-Dem v15 | ⚠️ 待手动下载 → 该分量降级为 0 | 指引见 §6 |
| `data/external/*.csv`（模拟栈） | **仅回归夹具** | `simulate.write_simulated_data()` | ⛔ 模拟数据；`data.external_source: simulated` 时才生效 | 脚本生成，SHA-256 已登记 |

> **A2 决策（2026-11-15，作者确认）**：管道默认 `data.external_source: real`，消费 `data/external_real/`（官方原件转换产物）。
> **论文全部数值以真实数据栈重跑为准**（见 §8）；模拟栈仅作为可复现性回归夹具保留。

---

## 2. 军费与宏观经济数据（真实）

- **API**：`https://api.worldbank.org/v2/country/all/indicator/MS.MIL.XPND.CD`，`src/conflict_ew/data_fetch.py`，抓取时间戳写入 `data/raw_real.json → meta.fetched_at`（本次快照：2026-08-17，1970–2024）。
- **口径声明（决策 A）**：论文侧统一改口为 *"World Bank WDI series MS.MIL.XPND.CD (SIPRI-reported military expenditure, current USD)"*；不再声称直接引用 SIPRI Database 2025。
- **WDI ↔ SIPRI 抽核**（`scripts/verify_wdi_sipri.py`，官方原件 `raw/SIPRI-Milex-data-1949-2025_v1.2.xlsx`，SHA-256 `6cc3a30b…`）：
  - 抽核 100 国-年：**中位相对差 0，86% < 1%，65/100 完全相同**；
  - 残差集中于 SIPRI 高频修订国（ISR/CAN/ZAF）与近年年份——性质为**数据库版本差**，非口径差。
  - 结论：WDI 转发口径成立，论文措辞修改后与仓库一致。

## 3. ⛔ 历史披露：原 `data/external/*.csv` 为模拟数据（审查项 P0-7，已由 A2 决策修复）

**事实**：截至 2026-11-14，仓库管道消费的六个外部文件由
`src/conflict_ew/simulate.py::write_simulated_data()` 按面板骨架（8,505 行，每国每年无缺口）生成：

| 文件 | SHA-256（前 16 位） | 真实来源应为 |
|---|---|---|
| cow_conflicts.csv | `65979175af8813f1` | COW Intra-State Wars v5.1 |
| polity_vdem.csv | `8dba2d30afd7ce25` | Polity5 + V-Dem |
| icow.csv | `741757b64d6723eb` | ICOW Territorial Claims |
| atop.csv | `80ffbb75cd75834d` | ATOP 5.1 |
| cow_adjacency.csv | `4fb94a20339c9173` | COW Direct Contiguity v3.2 |
| countries.csv | `e6e06c7697917c1e` | COW state list（手工编码表） |

**证据**（与官方原件比对）：
1. **Polity**：官方 Polity5 中 CHN 2000–2010 `polity2` 恒为 **−7**、RUS 2000–2004 恒为 **6**；仓库 `polity_vdem.csv` 同期为 −10~+10 逐年随机跳动。V-Dem LDI 列亦为均匀随机数。
2. **行数**：四个外部文件行数完全相同（=面板行数+表头），而真实 ICOW 只覆盖主张涉入国、真实 ATOP 为同盟 dyadic 数据——完整性形态按面板填充。
3. **生成代码**：`simulate.py` L430–435 可复现写出这四个文件。

**影响范围（历史）**：2026-11-14 之前的论文 golden 值（AUC 0.5379 等）建立在此模拟栈上。**A2 决策后管道已切换真实数据并全量重跑**（见 §8）；模拟文件保留仅为回归夹具，`config.data.external_source: simulated` 显式开启才生效。

## 4. ✅ 官方原件（`data/external/raw/`，自动下载）

| 文件 | 来源 URL | 版本 | SHA-256（完整） | 大小 |
|---|---|---|---|---|
| p5v2018.xls | `http://www.systemicpeace.org/inscr/p5v2018.xls` | Polity5 (polity2) | `f81248561cb4fd884b0a439c7c0af298bc7f81b993bcd42e006988cb6d37a71c` | 4,487,680 B |
| Intra-State-Wars-v5.1.zip | `https://correlatesofwar.org/wp-content/uploads/Intra-State-Wars-v5.1.zip` | v5.1 | `b2d1f4892d3cbb02604cf8e506c5c2c3e87fb85df75276f69b1089e67fb25236` | 887,487 B |
| DirectContiguity320.zip | `https://correlatesofwar.org/wp-content/uploads/DirectContiguity320.zip` | v3.2 | `f1be081934f353f395bb873093b695ea01d7dfcb48e89146bd984c45342ba17a` | 636,368 B |
| SIPRI-Milex-data-1949-2025_v1.2.xlsx | sipri.org（人工下载，用于 §2 抽核） | 2025 v1.2 | `6cc3a30b1064f9f02e60236667eef82e08cad42910ce630909d004ad2c398a9d` | — |

下载日期 2026-11-15；**重复执行 `scripts/fetch_external.py` 三源 SHA-256 逐字节一致**（复现性验收点）。

## 5. 转换产物（`data/external_real/`，官方原件 → 面板 schema）

由 `scripts/fetch_external.py --convert` 生成，经 `data.external_source: real` 接入管道：

| 产物 | 行数 | 转换规则 | 真实性抽检 |
|---|---|---|---|
| cow_conflicts.csv（**主标签**） | 7,774 国-年 | **UCDP intrastate onset v25.1**：onset=onset1（新冲突首发/间隔≥2年重燃）、active=conflict_ids 非空；gwno_a→ISO3 | 2015–2024 有 41 起 onset（埃塞 2019、亚美-阿塞 2020、俄 2023 等均与史实吻合） |
| cow_isw_country_years.csv（历史对照） | 875 国-战争年 | COW ISW v5.1；进行中战争（EndYr=-7/-8/-9）以覆盖末端 2015 删失 | AFG 1978 ✓、AGO ✓ |
| polity_vdem.csv | 6,667 行 / 149 国（1970–2018） | `polity2`；缺失码 −66/−77/−88→NaN；经 COW ccode→ISO3 | CHN 恒 −7、RUS 6、DEU 10、VEN 1999 后 8→5 ✓ |
| cow_adjacency.csv | 437 dyad | `contdir.csv` conttype≥1 且期间与 1970–2024 重叠 | — |
| countries.csv | 165 国 | 手工编码表（cow_code↔ISO3 映射基准，非外部数据源） | — |

**降级边界（真实模式当前生效）**：ICOW/ATOP/V-Dem 官方站不可脚本直连（§6）→ `F5_icow_claim_lag1`、`F6_atop_change_lag1`、`F4_vdem_ldi_change` 置 0 运行，run-log 以 `external_missing ... degraded=true` 标记；polity 2019 年后为 NaN（XGBoost 原生处理）。

## 6. ⚠️ 待手动下载（官方站禁止脚本直连）

| 源 | 指引 | 落盘路径 |
|---|---|---|
| ATOP 5.1 | 浏览器打开 https://www.atop-data.org/ → Data → dyadic CSV（Cloudflare 403） | `raw/atop_5_1.csv` |
| ICOW Territorial Claims | 浏览器打开 https://www.icow.org/（当前超时，需核对镜像） | `raw/icow_terrclaims.csv` |
| V-Dem v15 | https://www.v-dem.net/ 注册登录后下载 Country-Year CSV | `raw/vdem_cy.csv` |

手动放入后重跑 `python scripts/fetch_external.py --no-fetch`，脚本自动将其纳入 SHA-256 登记。

## 7. 复验方法

```bash
# 全量重抓（校验 SHA-256 与本文件 §4 一致）+ 登记 + 转换
python scripts/fetch_external.py --convert
# 离线重登记（不联网）
python scripts/fetch_external.py --no-fetch
# WDI↔SIPRI 军费抽核
python scripts/verify_wdi_sipri.py
```

## 8. 与论文整改的联动（R-11）

**✅ A2 决策已执行（2026-11-15）**：管道切换真实数据并全量重跑，golden 回归同步固化（`tests/test_golden_values.py` 等）。

1. **论文全部数值以新栈重跑值回写**（对照表见内部工作笔记，不入公开仓库）：
   - 主模型 AUC 0.5447 [0.4928, 0.5967]；Brier 0.0391→0.0370（isotonic）
   - 基线：LogReg(RE) 0.5716（配对差 −0.027，p=0.61 不显著）；VAR(2) 0.4479（+0.098，p=0.018 显著）
   - 消融：去 F1 → 0.6801（**军费异常信号在真实标签下为负贡献**）；仅 F1 → 0.4859
   - 测试窗 61 个真实正样本；lead-time 在原阈值定义下覆盖 0/61（指标需重新定义或如实报告退化）
2. **论文标签口径必须改写**：由 "COW Intra-State Wars" 改为 **"UCDP Intrastate armed conflict onset (v25.1)"**，§III 标签定义同步（UCDP 25 battle-deaths / 国家内冲突口径），并披露 ICOW/ATOP/V-Dem 降级与 Polity5 覆盖至 2018。
3. TABLE IV / Data Availability：军费措辞按 §2（决策 A）；数据来源逐项指向本文件与 `data/external/provenance.json`。
4. ISO3 映射：论文改口 "mapping available in code repository"（决策 B）→ 指向 `data/external_real/countries.csv`。
5. 结论/摘要必须按新叙事重写：真实标签下"军费异常信号无增量预警价值"的负结果 + 简单稀有事件 logit 占优。
