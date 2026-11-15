"""pytest 共享配置：注入 src 路径，提供小规模测试数据 fixture。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


@pytest.fixture(scope="session")
def small_panel() -> pd.DataFrame:
    """构造 3 国 × 25 年的小面板（含军费/协变量/外部变量/标签）。"""
    rng = np.random.default_rng(42)
    years = list(range(1990, 2015))
    rows = []
    for iso, base in (("AAA", 20.0), ("BBB", 21.0), ("CCC", 22.0)):
        log_mil = np.array([base + 0.03 * (y - 1990) for y in years])
        log_mil[12:15] += 0.6  # 2002-2004 军费异常抬升
        for j, y in enumerate(years):
            rows.append({
                "iso3": iso, "year": y,
                "mil_expense_log": float(log_mil[j]),
                "gdp_growth": float(rng.normal(3, 1.5)),
                "inflation": float(abs(rng.normal(4, 2))),
                "unemployment": float(rng.normal(7, 2)),
                "polity_score": int(rng.integers(-5, 6)),
                "vdem_ldi": float(rng.uniform(0.1, 0.9)),
                "icow_claim": int(rng.random() < 0.15),
                "atop_change": int(rng.random() < 0.1),
                "onset": 0, "active": 0,
                "y_onset": 0,
            })
    # AAA 国 2003 年 onset
    df = pd.DataFrame(rows)
    mask = (df["iso3"] == "AAA") & (df["year"] == 2003)
    df.loc[mask, "onset"] = 1
    df.loc[mask, "y_onset"] = 1
    df.loc[(df["iso3"] == "AAA") & (df["year"] == 2004), "active"] = 1
    return df


@pytest.fixture(scope="session")
def small_features() -> pd.DataFrame:
    """6 因子特征表（3 国 × 25 年，含 y）。"""
    rng = np.random.default_rng(7)
    years = list(range(1990, 2015))
    rows = []
    for iso in ("AAA", "BBB", "CCC"):
        for y in years:
            rows.append({
                "iso3": iso, "year": y,
                "F1_e_star": float(rng.normal(0, 1)),
                "F2_arms_race": float(rng.normal(0, 0.1)),
                "F3_gdp_growth_lag1": float(rng.normal(3, 1.5)),
                "F3_inflation_lag1": float(abs(rng.normal(4, 2))),
                "F3_unemployment_lag1": float(rng.normal(7, 2)),
                "F4_polity_change": float(rng.integers(-2, 3)),
                "F4_vdem_ldi_change": float(rng.normal(0, 0.05)),
                "F5_icow_claim_lag1": float(rng.random() < 0.15),
                "F6_atop_change_lag1": float(rng.random() < 0.1),
                "y_onset": int((iso == "AAA") and (y == 2003)),
            })
    return pd.DataFrame(rows)
