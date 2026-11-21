"""tests/test_persistence.py：naive persistence 基线（冲突预测文献标准对照）。"""
import pytest

from conflict_ew import paths


def test_persistence_flags_lagged_active(small_panel, monkeypatch):
    """flag 规则 = 上一年 active；small_panel 仅 (AAA,2004) active=1 → 只标记 (AAA,2005)。"""
    cfg = paths.cfg()
    monkeypatch.setitem(cfg["model"]["cv"], "test_start", 2000)
    monkeypatch.setitem(cfg["model"]["cv"], "test_end", 2014)

    from conflict_ew.baselines import baseline_persistence

    features = small_panel[["iso3", "year", "y_onset"]]
    b = baseline_persistence(features, small_panel)

    n_window = int(((features["year"] >= 2000) & (features["year"] <= 2014)).sum())
    assert b["n"] == n_window  # 窗内每行均有上一年 lag（首年 1990 在窗外）
    assert b["flagged"] == 1   # 仅 (AAA,2005) 被标记
    assert b["TP"] == 0        # 窗内唯一 onset 在 AAA 2003，不被 persistence 覆盖
    assert 0.0 <= b["auc"] <= 1.0
    assert b["tau"] == 1.0 and b["tau_quantile"] is None
    assert b["precision_at_tau"] == pytest.approx(0.0)
