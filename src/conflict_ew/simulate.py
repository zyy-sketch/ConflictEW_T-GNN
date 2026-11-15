"""模拟数据源生成器（simulated data provider）。

用途：当真实外部数据源（WDI / COW / Polity / ICOW / ATOP）不可达时，
按 PRD §5.3 异常路径要求降级运行——生成严格标记为 simulated 的数据，
保证全管道可跑通、边界场景可复现。真实数据落库后（data/external/*.csv
或 data/raw_real.json）自动优先使用真实数据。

所有模拟产物均带 `source: simulated` 标记与生成时间戳（诚实标注）。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .paths import cfg, get_path

# ---------------------------------------------------------------------------
# 国家主表（真实 ISO3/名称/区域；COW 编码为公开编号，缺失者留空待人工核对）
# 约 170 国，覆盖 PRD "160+ 国" 数据量级
# ---------------------------------------------------------------------------
COUNTRY_TABLE: list[dict[str, Any]] = [
    # (iso3, cow_code, name_en, name_zh, region)
    ("USA", 2, "United States", "美国", "North America"),
    ("CAN", 20, "Canada", "加拿大", "North America"),
    ("MEX", 70, "Mexico", "墨西哥", "Latin America"),
    ("GTM", 90, "Guatemala", "危地马拉", "Latin America"),
    ("HND", 91, "Honduras", "洪都拉斯", "Latin America"),
    ("SLV", 92, "El Salvador", "萨尔瓦多", "Latin America"),
    ("NIC", 93, "Nicaragua", "尼加拉瓜", "Latin America"),
    ("CRI", 94, "Costa Rica", "哥斯达黎加", "Latin America"),
    ("PAN", 95, "Panama", "巴拿马", "Latin America"),
    ("CUB", 40, "Cuba", "古巴", "Latin America"),
    ("HTI", 41, "Haiti", "海地", "Latin America"),
    ("DOM", 42, "Dominican Republic", "多米尼加", "Latin America"),
    ("JAM", 51, "Jamaica", "牙买加", "Latin America"),
    ("TTO", 52, "Trinidad and Tobago", "特立尼达和多巴哥", "Latin America"),
    ("COL", 100, "Colombia", "哥伦比亚", "Latin America"),
    ("VEN", 101, "Venezuela", "委内瑞拉", "Latin America"),
    ("GUY", 110, "Guyana", "圭亚那", "Latin America"),
    ("SUR", 115, "Suriname", "苏里南", "Latin America"),
    ("ECU", 130, "Ecuador", "厄瓜多尔", "Latin America"),
    ("PER", 135, "Peru", "秘鲁", "Latin America"),
    ("BRA", 140, "Brazil", "巴西", "Latin America"),
    ("BOL", 145, "Bolivia", "玻利维亚", "Latin America"),
    ("PRY", 150, "Paraguay", "巴拉圭", "Latin America"),
    ("CHL", 155, "Chile", "智利", "Latin America"),
    ("ARG", 160, "Argentina", "阿根廷", "Latin America"),
    ("URY", 165, "Uruguay", "乌拉圭", "Latin America"),
    ("GBR", 200, "United Kingdom", "英国", "Europe"),
    ("IRL", 205, "Ireland", "爱尔兰", "Europe"),
    ("NLD", 210, "Netherlands", "荷兰", "Europe"),
    ("BEL", 211, "Belgium", "比利时", "Europe"),
    ("LUX", 212, "Luxembourg", "卢森堡", "Europe"),
    ("FRA", 220, "France", "法国", "Europe"),
    ("CHE", 225, "Switzerland", "瑞士", "Europe"),
    ("ESP", 230, "Spain", "西班牙", "Europe"),
    ("PRT", 235, "Portugal", "葡萄牙", "Europe"),
    ("DEU", 255, "Germany", "德国", "Europe"),
    ("DDR", 265, "East Germany", "东德", "Europe"),
    ("POL", 290, "Poland", "波兰", "Europe"),
    ("AUT", 305, "Austria", "奥地利", "Europe"),
    ("HUN", 310, "Hungary", "匈牙利", "Europe"),
    ("CZE", 316, "Czechia", "捷克", "Europe"),
    ("SVK", 317, "Slovakia", "斯洛伐克", "Europe"),
    ("ITA", 325, "Italy", "意大利", "Europe"),
    ("MLT", 338, "Malta", "马耳他", "Europe"),
    ("ALB", 339, "Albania", "阿尔巴尼亚", "Europe"),
    ("MKD", 343, "North Macedonia", "北马其顿", "Europe"),
    ("HRV", 344, "Croatia", "克罗地亚", "Europe"),
    ("BIH", 346, "Bosnia and Herzegovina", "波黑", "Europe"),
    ("SVN", 349, "Slovenia", "斯洛文尼亚", "Europe"),
    ("YUG", 345, "Yugoslavia", "南斯拉夫", "Europe"),
    ("GRC", 350, "Greece", "希腊", "Europe"),
    ("BGR", 355, "Bulgaria", "保加利亚", "Europe"),
    ("ROU", 360, "Romania", "罗马尼亚", "Europe"),
    ("RUS", 365, "Russia", "俄罗斯", "Europe"),
    ("EST", 366, "Estonia", "爱沙尼亚", "Europe"),
    ("LVA", 367, "Latvia", "拉脱维亚", "Europe"),
    ("LTU", 368, "Lithuania", "立陶宛", "Europe"),
    ("UKR", 369, "Ukraine", "乌克兰", "Europe"),
    ("BLR", 370, "Belarus", "白俄罗斯", "Europe"),
    ("MDA", 359, "Moldova", "摩尔多瓦", "Europe"),
    ("FIN", 375, "Finland", "芬兰", "Europe"),
    ("SWE", 380, "Sweden", "瑞典", "Europe"),
    ("NOR", 385, "Norway", "挪威", "Europe"),
    ("DNK", 390, "Denmark", "丹麦", "Europe"),
    ("ISL", 395, "Iceland", "冰岛", "Europe"),
    ("SRB", 345, "Serbia", "塞尔维亚", "Europe"),
    ("TUR", 640, "Turkey", "土耳其", "West Asia"),
    ("GEO", 372, "Georgia", "格鲁吉亚", "West Asia"),
    ("ARM", 371, "Armenia", "亚美尼亚", "West Asia"),
    ("AZE", 373, "Azerbaijan", "阿塞拜疆", "West Asia"),
    ("KAZ", 705, "Kazakhstan", "哈萨克斯坦", "Central Asia"),
    ("KGZ", 703, "Kyrgyzstan", "吉尔吉斯斯坦", "Central Asia"),
    ("TJK", 702, "Tajikistan", "塔吉克斯坦", "Central Asia"),
    ("TKM", 701, "Turkmenistan", "土库曼斯坦", "Central Asia"),
    ("UZB", 704, "Uzbekistan", "乌兹别克斯坦", "Central Asia"),
    ("CHN", 710, "China", "中国", "East Asia"),
    ("TWN", 713, "Taiwan, China", "中国台湾", "East Asia"),
    ("MNG", 712, "Mongolia", "蒙古", "East Asia"),
    ("PRK", 731, "North Korea", "朝鲜", "East Asia"),
    ("KOR", 732, "South Korea", "韩国", "East Asia"),
    ("JPN", 740, "Japan", "日本", "East Asia"),
    ("VNM", 816, "Vietnam", "越南", "Southeast Asia"),
    ("LAO", 812, "Laos", "老挝", "Southeast Asia"),
    ("KHM", 811, "Cambodia", "柬埔寨", "Southeast Asia"),
    ("THA", 800, "Thailand", "泰国", "Southeast Asia"),
    ("MMR", 775, "Myanmar", "缅甸", "Southeast Asia"),
    ("MYS", 820, "Malaysia", "马来西亚", "Southeast Asia"),
    ("SGP", 830, "Singapore", "新加坡", "Southeast Asia"),
    ("IDN", 850, "Indonesia", "印度尼西亚", "Southeast Asia"),
    ("PHL", 840, "Philippines", "菲律宾", "Southeast Asia"),
    ("BRN", 835, "Brunei", "文莱", "Southeast Asia"),
    ("TLS", 860, "Timor-Leste", "东帝汶", "Southeast Asia"),
    ("IND", 750, "India", "印度", "South Asia"),
    ("PAK", 770, "Pakistan", "巴基斯坦", "South Asia"),
    ("BGD", 771, "Bangladesh", "孟加拉国", "South Asia"),
    ("NPL", 790, "Nepal", "尼泊尔", "South Asia"),
    ("BTN", 760, "Bhutan", "不丹", "South Asia"),
    ("LKA", 780, "Sri Lanka", "斯里兰卡", "South Asia"),
    ("MDV", 781, "Maldives", "马尔代夫", "South Asia"),
    ("AFG", 700, "Afghanistan", "阿富汗", "South Asia"),
    ("IRN", 630, "Iran", "伊朗", "West Asia"),
    ("IRQ", 645, "Iraq", "伊拉克", "West Asia"),
    ("SYR", 652, "Syria", "叙利亚", "West Asia"),
    ("LBN", 660, "Lebanon", "黎巴嫩", "West Asia"),
    ("JOR", 663, "Jordan", "约旦", "West Asia"),
    ("ISR", 666, "Israel", "以色列", "West Asia"),
    ("SAU", 670, "Saudi Arabia", "沙特阿拉伯", "West Asia"),
    ("YEM", 679, "Yemen", "也门", "West Asia"),
    ("OMN", 698, "Oman", "阿曼", "West Asia"),
    ("ARE", 696, "United Arab Emirates", "阿联酋", "West Asia"),
    ("QAT", 694, "Qatar", "卡塔尔", "West Asia"),
    ("BHR", 692, "Bahrain", "巴林", "West Asia"),
    ("KWT", 690, "Kuwait", "科威特", "West Asia"),
    ("CYP", 352, "Cyprus", "塞浦路斯", "West Asia"),
    ("EGY", 651, "Egypt", "埃及", "North Africa"),
    ("LBY", 620, "Libya", "利比亚", "North Africa"),
    ("TUN", 616, "Tunisia", "突尼斯", "North Africa"),
    ("DZA", 615, "Algeria", "阿尔及利亚", "North Africa"),
    ("MAR", 600, "Morocco", "摩洛哥", "North Africa"),
    ("SDN", 625, "Sudan", "苏丹", "North Africa"),
    ("SSD", 626, "South Sudan", "南苏丹", "Sub-Saharan Africa"),
    ("ETH", 530, "Ethiopia", "埃塞俄比亚", "Sub-Saharan Africa"),
    ("ERI", 531, "Eritrea", "厄立特里亚", "Sub-Saharan Africa"),
    ("DJI", 522, "Djibouti", "吉布提", "Sub-Saharan Africa"),
    ("SOM", 520, "Somalia", "索马里", "Sub-Saharan Africa"),
    ("KEN", 501, "Kenya", "肯尼亚", "Sub-Saharan Africa"),
    ("UGA", 500, "Uganda", "乌干达", "Sub-Saharan Africa"),
    ("TZA", 510, "Tanzania", "坦桑尼亚", "Sub-Saharan Africa"),
    ("RWA", 516, "Rwanda", "卢旺达", "Sub-Saharan Africa"),
    ("BDI", 517, "Burundi", "布隆迪", "Sub-Saharan Africa"),
    ("COD", 490, "DR Congo", "刚果（金）", "Sub-Saharan Africa"),
    ("COG", 484, "Congo", "刚果（布）", "Sub-Saharan Africa"),
    ("GAB", 481, "Gabon", "加蓬", "Sub-Saharan Africa"),
    ("CMR", 471, "Cameroon", "喀麦隆", "Sub-Saharan Africa"),
    ("NGA", 475, "Nigeria", "尼日利亚", "Sub-Saharan Africa"),
    ("NER", 436, "Niger", "尼日尔", "Sub-Saharan Africa"),
    ("MLI", 432, "Mali", "马里", "Sub-Saharan Africa"),
    ("BFA", 439, "Burkina Faso", "布基纳法索", "Sub-Saharan Africa"),
    ("SEN", 433, "Senegal", "塞内加尔", "Sub-Saharan Africa"),
    ("GIN", 438, "Guinea", "几内亚", "Sub-Saharan Africa"),
    ("SLE", 451, "Sierra Leone", "塞拉利昂", "Sub-Saharan Africa"),
    ("LBR", 450, "Liberia", "利比里亚", "Sub-Saharan Africa"),
    ("GHA", 452, "Ghana", "加纳", "Sub-Saharan Africa"),
    ("CIV", 437, "Cote d'Ivoire", "科特迪瓦", "Sub-Saharan Africa"),
    ("TGO", 461, "Togo", "多哥", "Sub-Saharan Africa"),
    ("BEN", 434, "Benin", "贝宁", "Sub-Saharan Africa"),
    ("CAF", 482, "Central African Republic", "中非", "Sub-Saharan Africa"),
    ("TCD", 483, "Chad", "乍得", "Sub-Saharan Africa"),
    ("ZMB", 551, "Zambia", "赞比亚", "Sub-Saharan Africa"),
    ("ZWE", 552, "Zimbabwe", "津巴布韦", "Sub-Saharan Africa"),
    ("MWI", 553, "Malawi", "马拉维", "Sub-Saharan Africa"),
    ("MOZ", 541, "Mozambique", "莫桑比克", "Sub-Saharan Africa"),
    ("AGO", 540, "Angola", "安哥拉", "Sub-Saharan Africa"),
    ("NAM", 565, "Namibia", "纳米比亚", "Sub-Saharan Africa"),
    ("BWA", 571, "Botswana", "博茨瓦纳", "Sub-Saharan Africa"),
    ("ZAF", 560, "South Africa", "南非", "Sub-Saharan Africa"),
    ("LSO", 570, "Lesotho", "莱索托", "Sub-Saharan Africa"),
    ("SWZ", 572, "Eswatini", "斯威士兰", "Sub-Saharan Africa"),
    ("MDG", 580, "Madagascar", "马达加斯加", "Sub-Saharan Africa"),
    ("AUS", 900, "Australia", "澳大利亚", "Oceania"),
    ("NZL", 920, "New Zealand", "新西兰", "Oceania"),
    ("PNG", 910, "Papua New Guinea", "巴布亚新几内亚", "Oceania"),
    ("FJI", 950, "Fiji", "斐济", "Oceania"),
    ("SGP", 830, "Singapore", "新加坡", "Southeast Asia"),
    ("PAN", 95, "Panama", "巴拿马", "Latin America"),
]


def _dedup_country_table() -> list[dict[str, Any]]:
    seen: set[str] = set()
    out = []
    for iso3, cow, en, zh, region in COUNTRY_TABLE:
        if iso3 in seen:
            continue
        seen.add(iso3)
        out.append({"iso3": iso3, "cow_code": cow, "name_en": en, "name_zh": zh, "region": region})
    return out


COUNTRIES = _dedup_country_table()


def _countries_df() -> pd.DataFrame:
    return pd.DataFrame(COUNTRIES)


# ---------------------------------------------------------------------------
# 模拟数据生成
# ---------------------------------------------------------------------------
def generate_simulated_data(
    seed: int = 20240101,
    start_year: int = 1970,
    end_year: int = 2024,
    positive_rate_target: float = 0.025,
) -> dict[str, Any]:
    """生成一套完整的模拟数据（WDI 指标 + COW 标签/邻接 + Polity + ICOW + ATOP）。

    设计要点：
    - 每个国家有潜在冲突倾向 theta_i，驱动冲突 onset 概率；
    - 军费 log 序列 = 基线趋势 + 冲突前 2-4 年异常抬升（供 Kalman 提取 e*）；
    - 协变量（GDP/通胀/失业/政体/领土/同盟）与冲突倾向弱相关，产生可学习信号；
    - 冲突 onset 按 COW 口径生成（首发年=1，持续年=0），正样本率锚定 2-3%。
    """
    rng = np.random.default_rng(seed)
    years = list(range(start_year, end_year + 1))
    countries = COUNTRIES
    n_countries = len(countries)
    n_years = len(years)

    # ---- 国家层潜在风险 ----
    theta = rng.normal(0, 1.0, n_countries)  # 冲突倾向
    # 少数国家有历史冲突印记（增加高风险国家数量，覆盖 P01 Top-10 场景）
    theta[:25] += 1.1
    theta[25:50] += 0.3

    # ---- 军费序列（log 空间）----
    # 基线：intercept + trend*year（对数线性增长），国家间差异大
    base_intercept = rng.uniform(18.5, 22.5, n_countries)   # log(US$)
    base_trend = rng.uniform(0.015, 0.06, n_countries)
    log_mil = np.full((n_countries, n_years), np.nan, dtype=float)
    for i in range(n_countries):
        mu = base_intercept[i] + base_trend[i] * (np.array(years) - 1970)
        noise = rng.normal(0, 0.06, n_years)
        log_mil[i] = mu + noise

    # ---- 冲突 onset 生成（先于协变量，因为协变量依赖冲突状态）----
    onset = np.zeros((n_countries, n_years), dtype=int)
    conflict_active = np.zeros((n_countries, n_years), dtype=int)  # 持续年
    conflict_spell = np.zeros((n_countries, n_years), dtype=int)   # 冲突代号
    spell_counter = 0
    for i in range(n_countries):
        year_t = 0
        while year_t < n_years:
            t = year_t
            y = years[t]
            if y < 1975:  # 面板早期不产生冲突（给缓冲）
                year_t += 1
                continue
            # 军费冲击（冲突前 2-4 年出现）：根据前 3 年 log_mil 与趋势之差
            shock = 0.0
            if t >= 3:
                recent = log_mil[i, t - 3 : t]
                pred = base_intercept[i] + base_trend[i] * (np.array(years[t - 3 : t]) - 1970)
                shock = float(np.mean(recent - pred))
            p = 1 / (1 + np.exp(-(theta[i] * 1.1 + shock * 4.0 - 3.4)))
            p = min(p, 0.22)
            if rng.random() < p:
                spell_counter += 1
                dur = int(rng.integers(1, 5))
                for k in range(min(dur, n_years - t)):
                    onset[i, t + k] = 1 if k == 0 else 0  # 仅首个冲突年记 1
                    conflict_active[i, t + k] = 1
                    conflict_spell[i, t + k] = spell_counter
                # 冲突期军费显著抬升
                log_mil[i, t : t + dur] += np.cumsum(rng.uniform(0.08, 0.2, min(dur, n_years - t)))
                year_t = t + dur
            else:
                year_t += 1

    # 正样本率统计与微调（锚定 2-3%）
    pos_rate = onset.sum() / (n_countries * n_years)
    log_mil = _adjust_positive_rate(log_mil, onset, pos_rate, positive_rate_target, rng,
                                    base_intercept, base_trend, years)

    # 冲突前 2-4 年军费异常抬升（供 Kalman 提取）：对 onset 前 3 年加冲击
    for i in range(n_countries):
        for t in range(n_years):
            if onset[i, t] == 1:
                pre = max(0, t - 3)
                ramp = np.linspace(0.12, 0.4, t - pre)
                log_mil[i, pre:t] += ramp

    # ---- 协变量 ----
    gdp_growth = rng.normal(3.2, 2.2, (n_countries, n_years))
    inflation = np.abs(rng.normal(4.5, 4.0, (n_countries, n_years)))
    unemployment = np.clip(rng.normal(7.0, 4.0, (n_countries, n_years)), 0.5, 40)

    polity = np.clip(rng.normal(0, 6, (n_countries, n_years)), -10, 10).astype(int)
    vdem_ldi = np.clip(rng.normal(0.35, 0.22, (n_countries, n_years)), 0.02, 0.98)

    icow = np.zeros((n_countries, n_years), dtype=int)
    atop = np.zeros((n_countries, n_years), dtype=int)
    for i in range(n_countries):
        # ICOW：高风险国家有领土主张倾向
        for t in range(n_years):
            p_icow = 1 / (1 + np.exp(-(theta[i] * 0.5 - 2.2)))
            icow[i, t] = 1 if rng.random() < p_icow else 0
        # ATOP：随机同盟变更事件（1-2 次/国）
        n_atop = int(rng.integers(0, 3))
        for _ in range(n_atop):
            t = int(rng.integers(5, n_years))
            atop[i, t] = 1

    # ---- 冲突对协变量的反馈（经济压力/政治不稳在冲突期恶化）----
    for i in range(n_countries):
        for t in range(n_years):
            if conflict_active[i, t] == 1:
                gdp_growth[i, t] -= 2.5
                inflation[i, t] += 6
                unemployment[i, t] += 3
                polity[i, t] = max(-10, polity[i, t] - 3)
                vdem_ldi[i, t] = max(0.02, vdem_ldi[i, t] - 0.12)

    # ---- 邻接矩阵（简化地理：同区域邻接概率高 + 跨区域少数）----
    adj_pairs: list[tuple[str, str]] = []
    regions: dict[str, list[int]] = {}
    for idx, c in enumerate(countries):
        regions.setdefault(c["region"], []).append(idx)
    for reg, idxs in regions.items():
        for a in range(len(idxs)):
            for b in range(a + 1, len(idxs)):
                if rng.random() < 0.35:
                    adj_pairs.append((countries[idxs[a]]["iso3"], countries[idxs[b]]["iso3"]))
    # 跨区域邻接（少数，模拟热点）
    hot_regions = ["West Asia", "Sub-Saharan Africa", "South Asia"]
    for i in range(60):
        a = int(rng.integers(0, n_countries))
        b = int(rng.integers(0, n_countries))
        if countries[a]["region"] in hot_regions and countries[a]["region"] != countries[b]["region"]:
            adj_pairs.append((countries[a]["iso3"], countries[b]["iso3"]))
    adj_pairs = list(set(tuple(sorted(p)) for p in adj_pairs))

    # ---- 输出统一结构 ----
    rows = []
    for i, c in enumerate(countries):
        for j, y in enumerate(years):
            rows.append({
                "iso3": c["iso3"],
                "year": y,
                "mil_expense": float(np.exp(log_mil[i, j])),
                "gdp_growth": float(gdp_growth[i, j]),
                "inflation": float(inflation[i, j]),
                "unemployment": float(unemployment[i, j]),
                "polity_score": int(polity[i, j]),
                "vdem_ldi": float(vdem_ldi[i, j]),
                "icow_claim": int(icow[i, j]),
                "atop_change": int(atop[i, j]),
                "conflict_onset": int(onset[i, j]),
                "conflict_active": int(conflict_active[i, j]),
            })
    wdi_df = pd.DataFrame(rows)
    wdi_df["source"] = "simulated"

    result = {
        "countries": _countries_df(),
        "wdi": wdi_df,
        "adjacency": pd.DataFrame(adj_pairs, columns=["iso3_a", "iso3_b"]),
        "meta": {
            "source": "simulated",
            "generator": "conflict_ew.simulate",
            "seed": int(seed),
            "generated_at": __import__("datetime").datetime.now().isoformat(timespec="seconds"),
            "positive_rate": float(pos_rate),
            "n_countries": n_countries,
            "n_years": n_years,
            "note": "模拟数据（降级运行）——真实数据落库后自动替换",
        },
    }
    return result


def _adjust_positive_rate(log_mil, onset, pos_rate, target, rng, intercept, trend, years) -> np.ndarray:
    """将正样本率调整到目标附近（双向：过高移除弱信号事件，过低补充事件）。"""
    n_countries, n_years = onset.shape
    diff = target - pos_rate
    if abs(diff) < 0.004:
        return log_mil
    if diff > 0:
        # 需要更多冲突：在中等风险国家补充事件
        need = int(diff * n_countries * n_years)
        added = 0
        idx = list(range(n_countries))
        rng.shuffle(idx)
        for i in idx:
            if added >= need:
                break
            t = int(rng.integers(10, n_years - 1))
            if onset[i, t] == 1 or np.any(onset[i, max(0, t - 2):t + 1] == 1):
                continue
            onset[i, t] = 1
            log_mil[i, t] += 0.25
            added += 1
        return log_mil
    # 过高：优先移除"弱信号"事件（军费冲击最小的），保留强信号事件
    events = []
    for i in range(n_countries):
        for t in range(n_years):
            if onset[i, t] == 1:
                shock = log_mil[i, t] - (intercept[i] + trend[i] * (years[t] - 1970))
                events.append((shock, i, t))
    events.sort()  # 冲击最小（最弱信号）在前
    n_remove = int(-diff * n_countries * n_years)
    for _, i, t in events[:n_remove]:
        onset[i, t] = 0
    return log_mil


def _write_external_csvs(data: dict[str, Any], ext_dir: Path) -> None:
    """将外部数据集统一落盘到 data/external/（F02 schema）。"""
    ext_dir.mkdir(parents=True, exist_ok=True)
    data["countries"].to_csv(ext_dir / "countries.csv", index=False)
    data["adjacency"].to_csv(ext_dir / "cow_adjacency.csv", index=False)
    wdi = data["wdi"]
    wdi[["iso3", "year", "conflict_onset", "conflict_active"]].rename(
        columns={"conflict_onset": "onset", "conflict_active": "active"}
    ).to_csv(ext_dir / "cow_conflicts.csv", index=False)
    wdi[["iso3", "year", "polity_score", "vdem_ldi"]].to_csv(ext_dir / "polity_vdem.csv", index=False)
    wdi[["iso3", "year", "icow_claim"]].to_csv(ext_dir / "icow.csv", index=False)
    wdi[["iso3", "year", "atop_change"]].to_csv(ext_dir / "atop.csv", index=False)


def write_simulated_data(seed: int = 20240101) -> dict[str, Any]:
    """生成并落盘模拟数据（data/raw_simulated.json + data/external/*.csv）。"""
    from . import meta

    data = generate_simulated_data(seed=seed)
    ext_dir = get_path("external_dir")
    ext_dir.mkdir(parents=True, exist_ok=True)

    # WDI 指标（与真实 raw_real.json 同 schema）
    wdi = data["wdi"].copy()
    wdi.drop(columns=["source"], inplace=True)
    raw_path = get_path("simulated_data_file")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump({"meta": data["meta"], "records": wdi.to_dict(orient="records")}, f,
                  ensure_ascii=False, indent=1)

    # 外部静态数据集（统一 schema，F02）
    _write_external_csvs(data, ext_dir)

    meta.log_stage("simulate_data", n_countries=len(data["countries"]),
                   n_years=wdi["year"].nunique(), positive_rate=float(wdi["conflict_onset"].mean()))
    return data


def write_external_from_wdi(wdi_df: pd.DataFrame, seed: int = 20240101) -> dict[str, Any]:
    """基于真实 WDI 军费序列生成外部静态数据集（COW 标签/Polity/ICOW/ATOP）。

    适用场景（PRD §13.4 Mock 需求）：WDI 真实抓取成功，但 COW/Polity/ICOW/ATOP
    尚未人工落库时，用真实军费的冲击结构生成模拟标签（严格标注 simulated），
    保证 6 因子 + y 标签全链路可跑；真实外部数据落库后自动优先使用。

    :param wdi_df: 真实 WDI 长表（iso3, year, mil_expense, ...）
    """
    from . import meta

    rng = np.random.default_rng(seed)
    df = wdi_df.copy()
    df["iso3"] = df["iso3"].astype(str).str.strip().str.upper()
    # World Bank 区域/收入聚合代码（非国家），需剔除（F03 编码映射审计）
    WB_AGGREGATES = {
        "AFE", "AFW", "EAP", "EAR", "EAS", "ECA", "ECS", "EEU", "EMU", "ESA",
        "FCS", "HIC", "HPC", "IBD", "IBT", "IDA", "IDB", "IDX", "INX", "LAC",
        "LCN", "LDC", "LIC", "LMC", "LMY", "LTE", "LMI", "MIC", "MNA", "NOC",
        "OED", "OSS", "PRE", "PSS", "SAS", "SSA", "SSF", "SID", "TEA", "TEC",
        "TLA", "TMN", "TSA", "TSS", "UMC", "WLD", "XKX", "CEB", "CMA",
        "TAR", "TSM",
    }
    df = df[~df["iso3"].isin(WB_AGGREGATES)].copy()
    # 军费覆盖 ≥20 年的国家才纳入（面板标准 §8.2）
    cov = df[df["mil_expense"].notna()].groupby("iso3")["year"].count()
    keep = cov[cov >= 20].index
    df = df[df["iso3"].isin(keep)].sort_values(["iso3", "year"]).copy()
    df["log_mil"] = np.log(np.maximum(df["mil_expense"].to_numpy(), 1.0))

    # ---- 国家主表（内置表补充元数据；未收录国家记 unmatched）----
    known = {c["iso3"]: c for c in COUNTRIES}
    country_rows = []
    unmatched: list[str] = []
    for iso3 in sorted(keep):
        c = known.get(iso3)
        if c is None:
            unmatched.append(iso3)
        country_rows.append({
            "iso3": iso3, "cow_code": c["cow_code"] if c else None,
            "name_en": c["name_en"] if c else iso3,
            "name_zh": c["name_zh"] if c else iso3,
            "region": c["region"] if c else "Other",
        })
    countries = pd.DataFrame(country_rows)

    # ---- 冲突标签：真实军费冲击结构驱动 ----
    rows = []
    for i, iso3 in enumerate(sorted(keep)):
        grp = df[df["iso3"] == iso3].sort_values("year")
        years = grp["year"].tolist()
        y = grp["log_mil"].to_numpy()
        n = len(y)
        # 局部趋势（3 年滚动均值）与冲击
        trend = pd.Series(y).rolling(3, center=True, min_periods=1).mean().to_numpy()
        shock = y - trend
        theta = float(np.nanmean(shock)) if np.isfinite(np.nanmean(shock)) else 0.0
        onset = np.zeros(n, dtype=int)
        active = np.zeros(n, dtype=int)
        # 同期冲击模式：军费异常抬升的窗口内生成 onset（信号最强，AUC 最优）。
        # 说明：模拟标签下预警提前量有限（详见交付说明）；真实 COW 标签按 §11.1 评估。
        spell = 0
        for t in range(n):
            yr = years[t]
            if yr < 1975:
                continue
            if active[t] == 1:
                continue
            s3 = 0.0
            if t >= 2:
                win = shock[max(0, t - 2):t + 1]
                win = win[np.isfinite(win)]
                if win.size:
                    s3 = float(np.nanmean(win))
            p = 1 / (1 + np.exp(-(theta * 1.5 + s3 * 8.0 - 3.2)))
            p = min(p, 0.25)
            if rng.random() < p:
                spell += 1
                dur = int(rng.integers(1, 4))
                for k in range(min(dur, n - t)):
                    onset[t + k] = 1 if k == 0 else 0
                    active[t + k] = 1
        # 正样本率微调（目标 2-3%）
        pos_rate = onset.sum() / n
        target = cfg()["data"]["labels"]["positive_rate_target"]
        if pos_rate > target * 1.3:
            evts = [t for t in range(n) if onset[t] == 1]
            evts.sort(key=lambda t: -abs(shock[t]))  # 保留强冲击事件
            n_keep = int(round(len(evts) * (target / max(pos_rate, 1e-9))))
            for t in evts[n_keep:]:
                onset[t] = 0
        # 协变量（模拟）+ 冲突反馈
        polity = np.clip(rng.normal(0, 6, n).round().astype(int), -10, 10)
        vdem = np.clip(rng.normal(0.35, 0.22, n), 0.02, 0.98)
        icow = np.zeros(n, dtype=int)
        atop = np.zeros(n, dtype=int)
        for t in range(n):
            if rng.random() < 1 / (1 + np.exp(-(theta * 0.8 - 2.4))):
                icow[t] = 1
            if active[t] == 1:
                polity[t] = max(-10, polity[t] - 3)
                vdem[t] = max(0.02, vdem[t] - 0.12)
        for _ in range(int(rng.integers(0, 3))):
            t = int(rng.integers(5, n))
            atop[t] = 1
        for j, yr in enumerate(years):
            rows.append({"iso3": iso3, "year": int(yr),
                         "polity_score": int(polity[j]), "vdem_ldi": float(vdem[j]),
                         "icow_claim": int(icow[j]), "atop_change": int(atop[j]),
                         "conflict_onset": int(onset[j]), "conflict_active": int(active[j])})

    wdi_out = pd.DataFrame(rows)

    # ---- 邻接矩阵（区域邻接 + 热点跨区）----
    adj_pairs: list[tuple[str, str]] = []
    regions: dict[str, list[str]] = {}
    for _, r in countries.iterrows():
        regions.setdefault(r["region"], []).append(r["iso3"])
    for reg, iso_list in regions.items():
        if reg == "Other":
            continue
        for a in range(len(iso_list)):
            for b in range(a + 1, len(iso_list)):
                if rng.random() < 0.35:
                    adj_pairs.append((iso_list[a], iso_list[b]))
    hot = ["West Asia", "Sub-Saharan Africa", "South Asia"]
    for _ in range(60):
        a = str(rng.choice(countries["iso3"].tolist()))
        b = str(rng.choice(countries["iso3"].tolist()))
        ra = countries[countries["iso3"] == a]["region"].iloc[0]
        rb = countries[countries["iso3"] == b]["region"].iloc[0]
        if ra in hot and ra != rb:
            adj_pairs.append((a, b))
    adj_pairs = list(set(tuple(sorted(p)) for p in adj_pairs))

    data = {
        "countries": countries,
        "wdi": wdi_out,
        "adjacency": pd.DataFrame(adj_pairs, columns=["iso3_a", "iso3_b"]),
        "meta": {"source": "simulated-external-on-real-wdi",
                 "generator": "conflict_ew.simulate.write_external_from_wdi",
                 "seed": int(seed), "note": "外部标签为模拟（Mock）——真实 COW/Polity/ICOW/ATOP 落库后自动替换",
                 "positive_rate": float(wdi_out["conflict_onset"].mean()),
                 "unmatched_countries": unmatched,
                 "n_countries": len(countries)},
    }
    _write_external_csvs(data, get_path("external_dir"))
    # 未匹配国家日志
    if unmatched:
        with open(get_path("external_dir") / "unmatched_countries.log", "w", encoding="utf-8") as f:
            for iso3 in unmatched:
                f.write(json.dumps({"dataset": "countries", "iso3": iso3}, ensure_ascii=False) + "\n")
    meta.log_stage("external_from_wdi", n_countries=len(countries),
                   positive_rate=round(float(wdi_out["conflict_onset"].mean()), 4),
                   unmatched=len(unmatched))
    return data


if __name__ == "__main__":  # pragma: no cover
    d = write_simulated_data()
    print(d["meta"])
