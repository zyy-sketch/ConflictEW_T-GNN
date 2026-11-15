# PROVENANCE — 数据溯源与来源链登记

> 登记生成：`scripts/fetch_external.py`（manifest：`data/external/provenance.json`）
> 最后更新：2026-11-15
> 面板口径：40 国 × 1970–2024，军费 = WDI `MS.MIL.XPND.CD`（决策 A，2026-11-14 固化）

---

## 1. 总览

| 数据 | 角色 | 来源 | 状态 | 溯源等级 |
|---|---|---|---|---|
| 军费 / GDP 增速 / 通胀 / 失业率 | 主特征（F1–F3） | World Bank WDI API（实时抓取） | ✅ 真实，自动 | URL + 抓取时间戳 + SIPRI 抽核 |
| Polity 政体指数 | 协变量（F4） | Center for Systemic Peace, Polity5 | ✅ 真实，自动下载 | URL + 版本 + SHA-256 |
| 内战冲突标签 | **y 标签** | COW Intra-State Wars v5.1 | ✅ 真实，自动下载 | URL + 版本 + SHA-256 |
| 国家邻接 | 协变量（F2 邻国项） | COW Direct Contiguity v3.2 | ✅ 真实，自动下载 | URL + 版本 + SHA-256 |
| 同盟义务变化 | 协变量（F6） | ATOP 5.1 | ⚠️ 待手动下载 | 指引见 §6 |
| 领土主张 | 协变量（F5） | ICOW Territorial Claims | ⚠️ 待手动下载 | 指引见 §6 |
| V-Dem LDI | 协变量（F4） | V-Dem v15 | ⚠️ 待手动下载（需注册） | 指引见 §6 |
| **`data/external/*.csv`（现行管道读取）** | 全部外部列 + y 标签 | `src/conflict_ew/simulate.py::write_simulated_data()` | ⛔ **模拟数据（披露见 §3）** | 脚本生成，SHA-256 已登记 |

---

## 2. 军费与宏观经济数据（真实）

- **API**：`https://api.worldbank.org/v2/country/all/indicator/MS.MIL.XPND.CD`，`src/conflict_ew/data_fetch.py`，抓取时间戳写入 `data/raw_real.json → meta.fetched_at`（本次快照：2026-08-17，1970–2024）。
- **口径声明（决策 A）**：论文侧统一改口为 *"World Bank WDI series MS.MIL.XPND.CD (SIPRI-reported military expenditure, current USD)"*；不再声称直接引用 SIPRI Database 2025。
- **WDI ↔ SIPRI 抽核**（`scripts/verify_wdi_sipri.py`，官方原件 `raw/SIPRI-Milex-data-1949-2025_v1.2.xlsx`，SHA-256 `6cc3a30b…`）：
  - 抽核 100 国-年：**中位相对差 0，86% < 1%，65/100 完全相同**；
  - 残差集中于 SIPRI 高频修订国（ISR/CAN/ZAF）与近年年份——性质为**数据库版本差**，非口径差。
  - 结论：WDI 转发口径成立，论文措辞修改后与仓库一致。

## 3. ⛔ 诚实披露：现行 `data/external/*.csv` 为模拟数据（审查项 P0-7）

**事实**：仓库当前管道（`external.py → build_panel.py → factors.py`）读取的六个外部文件由
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

**影响范围**：`F4_polity_change` / `F4_vdem_ldi_change` / `F5_icow_claim_lag1` / `F6_atop_change_lag1` 四个特征、F2 的邻接矩阵输入，以及 **`y_onset` 标签（由模拟 cow_conflicts.csv 派生）**。
**因此：论文全部 golden 数值（AUC 0.5379 等）建立在"真实 WDI 军费 + 模拟外部协变量/标签"的数据栈之上。**
论文回写（R-11）必须与这一事实对齐：或在论文中如实说明外部协变量为模拟占位（方法学演示定位），或在接入真实标签后重跑全部数值。两者禁止混用口径。

## 4. ✅ 官方原件（`data/external/raw/`，自动下载）

| 文件 | 来源 URL | 版本 | SHA-256（完整） | 大小 |
|---|---|---|---|---|
| p5v2018.xls | `http://www.systemicpeace.org/inscr/p5v2018.xls` | Polity5 (polity2) | `f81248561cb4fd884b0a439c7c0af298bc7f81b993bcd42e006988cb6d37a71c` | 4,487,680 B |
| Intra-State-Wars-v5.1.zip | `https://correlatesofwar.org/wp-content/uploads/Intra-State-Wars-v5.1.zip` | v5.1 | `b2d1f4892d3cbb02604cf8e506c5c2c3e87fb85df75276f69b1089e67fb25236` | 887,487 B |
| DirectContiguity320.zip | `https://correlatesofwar.org/wp-content/uploads/DirectContiguity320.zip` | v3.2 | `f1be081934f353f395bb873093b695ea01d7dfcb48e89146bd984c45342ba17a` | 636,368 B |
| SIPRI-Milex-data-1949-2025_v1.2.xlsx | sipri.org（人工下载，用于 §2 抽核） | 2025 v1.2 | `6cc3a30b1064f9f02e60236667eef82e08cad42910ce630909d004ad2c398a9d` | — |

下载日期 2026-11-15；**重复执行 `scripts/fetch_external.py` 三源 SHA-256 逐字节一致**（复现性验收点）。

## 5. 转换产物（`data/external_real/`，官方原件 → 面板 schema）

由 `scripts/fetch_external.py --convert` 生成，**不覆盖** `data/external/`（保护 golden 复现基准）：

| 产物 | 行数 | 转换规则 | 真实性抽检 |
|---|---|---|---|
| polity.csv | 6,667 行 / 149 国（1970–2018） | `polity2`；缺失码 −66/−77/−88→NaN；经 COW ccode→ISO3 | CHN 恒 −7、RUS 6、DEU 10、VEN 1999 后 8→5 ✓ |
| cow_conflicts.csv | 819 国-战争年 / 211 起 onset | 参战国表逐段期间（StartYr/EndYr k=1..4）展开；onset=战争首年 | AFG 1978 四月革命 onset ✓、AGO 内战持续 ✓ |
| cow_adjacency.csv | 437 dyad | `contdir.csv` conttype≥1 且期间与 1970–2024 重叠 | — |

**已知边界**：① 战争仍在进行时 `EndYr=−9` 的截尾段被丢弃（个别 onset 晚一年，如 AGO 1975→1976）；② 103 个未匹配实体（非国家行为体/已消亡国家编码）登记于 `external_real/unmatched.log`；③ `vdem_ldi` 列缺失（V-Dem 需手动获取，§6）。

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

1. TABLE IV / Data Availability：军费措辞按 §2（决策 A）；外部数据源表述必须与 §3 披露一致。
2. 若论文主张使用真实 Polity/ICOW/ATOP/COW 标签 → 必须先切换管道至 `data/external_real/`（含标签）并**重跑全部数值**（含 Table III、消融、lead time），golden 断言同步更新。
3. ISO3 映射：论文改口 "mapping available in code repository"（决策 B）→ 指向 `data/external/countries.csv`。
