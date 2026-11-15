"""tests/test_lookahead.py：防泄漏自检（PRD §8.2 前瞻性约束 / R9 / §13.4 新增）。

L1：e* 只用 ≤t 信息（Kalman filtered 残差，非平滑）
L2：经济/领土/同盟指标用 t-1 年值
L3：邻国因子用 t-1 增速差（F2_arms_race）
L4：Kalman 超参为全序列 MLE——论文已披露的例外（§III-A），不作泄漏断言；
    其影响由敏感性分析量化（scripts/09_window_sensitivity.py，M2：r=0.926 / flag 一致率 98.9%）
L5：blocked CV 按时间切分，测试期仅访问一次
"""
import numpy as np
import pandas as pd
import pytest

from conflict_ew.factors import build_features
from conflict_ew.kalman import _kalman_filter, compute_kalman_signals


def test_kalman_filtered_residual_uses_only_past():
    """L1：在 t 时刻计算的一步预测残差不使用 t 之后的数据。

    验证方法：用完整序列计算残差序列，与"逐点前缀序列"计算的残差对比——
    filtered 残差应与逐点计算一致（平滑残差会不同）。
    """
    rng = np.random.default_rng(0)
    y = 20 + 0.02 * np.arange(60) + rng.normal(0, 0.05, 60)
    s2e, s2n, s2z = 0.005, 0.001, 0.0005
    e_full, _, _, _, _ = _kalman_filter(y, s2e, s2n, s2z)
    e_prefix = []
    for t in range(1, len(y) + 1):
        e_t, _, _, _, _ = _kalman_filter(y[:t], s2e, s2n, s2z)
        e_prefix.append(e_t[-1])
    e_prefix = np.array(e_prefix)
    # filtered 残差逐点等价 → 不使用未来信息
    assert np.allclose(e_full, e_prefix, atol=1e-8)


def test_economic_factors_lagged(small_panel):
    """L2：经济因子必须为 t-1 滞后值。"""
    panel = small_panel[["iso3", "year", "mil_expense_log"]].copy()
    sig = compute_kalman_signals(panel)
    feats = build_features(small_panel, sig, None, out_path=None)
    feats = feats.sort_values(["iso3", "year"]).reset_index(drop=True)
    # F3 使用 lag1：比较面板 t-1 年 gdp_growth 与特征 t 年 F3_gdp_growth_lag1
    base = small_panel.sort_values(["iso3", "year"])
    for iso in ("AAA", "BBB", "CCC"):
        g = base[base["iso3"] == iso].reset_index(drop=True)
        f = feats[feats["iso3"] == iso].reset_index(drop=True)
        for i in range(1, len(g) - 1):
            assert f.loc[i, "F3_gdp_growth_lag1"] == pytest.approx(
                g.loc[i - 1, "gdp_growth"]), "经济因子必须滞后 1 年"
        # F5/F6 同理
        assert f.loc[1, "F5_icow_claim_lag1"] == pytest.approx(g.loc[0, "icow_claim"])
        assert f.loc[1, "F6_atop_change_lag1"] == pytest.approx(g.loc[0, "atop_change"])


def test_neighbor_factor_uses_lagged_growth(small_panel):
    """L3：F2 军备竞赛因子在 t 年只使用 ≤t-1 的增速信息（邻国 t-1 增速 − 本国 t-1 增速）。"""
    adjacency = pd.DataFrame({"iso3_a": ["AAA", "BBB"], "iso3_b": ["BBB", "CCC"]})
    sig = compute_kalman_signals(small_panel[["iso3", "year", "mil_expense_log"]].copy())
    feats = build_features(small_panel, sig, adjacency, out_path=None)
    feats = feats.sort_values(["iso3", "year"]).reset_index(drop=True)

    log_mil = small_panel.sort_values(["iso3", "year"]).set_index(["iso3", "year"])["mil_expense_log"]

    def growth(iso: str, y: int) -> float:
        """y 年相对 y-1 年的军费增速（log 差）。"""
        return float(log_mil.loc[(iso, y)]) - float(log_mil.loc[(iso, y - 1)])

    adjacency_map = {"AAA": ["BBB"], "BBB": ["CCC"], "CCC": []}
    isos = ("AAA", "BBB", "CCC")
    for iso, nbrs in adjacency_map.items():
        f = feats[feats["iso3"] == iso].reset_index(drop=True)
        for y in range(1992, 2015):  # mil_growth_lag1 自 1992 年起有定义
            got = f.loc[f["year"] == y, "F2_arms_race"].item()
            if nbrs:
                expected = max(growth(n, y - 1) for n in nbrs) - growth(iso, y - 1)
            else:
                # 无邻国降级：全球中位增速差近似（同样只依赖 ≤t-1 增速）
                median_g = float(np.median([growth(i, y - 1) for i in isos]))
                expected = median_g - growth(iso, y - 1)
            assert got == pytest.approx(expected), f"{iso} {y}: F2 必须只用 t-1 增速差"


def test_cv_windows_respect_temporal_order():
    """L5：blocked expanding-window 切分：训练截止 < 验证起点 < 测试 2015。"""
    from conflict_ew.xgb import _build_windows

    windows = _build_windows(1990, 2014, 5)
    for w in windows[:-1]:
        assert w["train_end"] < w["val_start"] <= w["val_end"]
        assert w["val_end"] <= 2014
    assert windows[-1]["fold"] == "full_train"
    assert windows[-1]["train_end"] == 2014
    # 训练截止严格小于测试起点
    assert all(w["train_end"] < 2015 for w in windows)
