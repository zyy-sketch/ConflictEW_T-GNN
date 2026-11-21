"""tests/test_external_convert_contract.py：官方原件→标签数据的离线契约测试。

直接调用 fetch_external 的转换函数（不下载、不写盘），锁定标签数据的
schema 与完整性不变量——转换逻辑回归（如 UCDP 文件列名/口径变更）在此即被
捕获，而不必等到下游 AUC 异动才发现。

前置：data/external/raw/ 下的官方原件随仓库提交（离线可跑）。
"""
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from fetch_external import _ccode_to_iso3, convert_cow_conflicts, convert_ucdp_labels  # noqa: E402


@pytest.fixture(scope="module")
def c2i() -> dict[int, str]:
    return _ccode_to_iso3()


def test_ucdp_labels_contract(c2i: dict[int, str]) -> None:
    """UCDP onset v25.1 → 标签表：schema/唯一性/取值域/年份范围。"""
    df: pd.DataFrame = convert_ucdp_labels(c2i, set())
    assert {"iso3", "year", "onset", "active"} <= set(df.columns)
    assert not df.duplicated(["iso3", "year"]).any(), "标签表存在 (iso3, year) 重复"
    assert df["onset"].isin([0, 1]).all(), "onset 必须是 0/1"
    assert df["active"].isin([0, 1]).all()
    assert df["iso3"].notna().all() and (df["iso3"] != "").all()
    assert df["year"].between(1946, 2026).all()
    # 面板标签源非空：正样本必须存在（否则标签转换已静默失效）
    assert int(df["onset"].sum()) > 0, "UCDP 转换结果 onset 全为 0——标签源疑似失效"


def test_cow_conflicts_contract(c2i: dict[int, str]) -> None:
    """COW Intra-State Wars → 冲突历史表：schema/唯一性/年份下界。"""
    df: pd.DataFrame = convert_cow_conflicts(c2i, set())
    assert {"iso3", "year", "onset", "active"} <= set(df.columns)
    assert not df.duplicated(["iso3", "year"]).any()
    assert df["iso3"].notna().all()
    assert (df["year"] >= 1970).all()  # 面板下界（config.panel.start_year）
