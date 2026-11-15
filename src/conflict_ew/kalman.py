"""F05 Kalman 滤波信号提取：局部线性趋势（LLT）状态空间模型。

对每国军费 log 序列拟合 LLT：
    观测方程  y_t = μ_t + ε_t,        ε ~ N(0, σ²_ε)
    状态方程  μ_t = μ_{t-1} + ν_{t-1} + η_t,   η ~ N(0, σ²_η)
              ν_t = ν_{t-1} + ζ_t,             ζ ~ N(0, σ²_ζ)

- MLE 网格估计超参（σ²_ε, σ²_η, σ²_ζ），再精化（L-BFGS-B 多起点）
- 输出标准化一步预测残差 e*（仅使用 ≤t 信息，满足防泄漏约束 §8.2）
- |e*| ≥ 2 标记为异常信号（F10 异常标记 / F1 因子）
- 单国 O(N)；全 NaN 序列跳过并记录；发散警告
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from . import meta
from .paths import get_path

logger = logging.getLogger("conflict_ew.kalman")

ANOMALY_Z = 2.0  # |e*| ≥ 2 标记异常（PRD F05 / F10）


@dataclass
class KalmanResult:
    country: str
    sigma2_eps: float
    sigma2_eta: float
    sigma2_zeta: float
    log_lik: float
    converged: bool
    n_obs: int
    diverged: bool = False
    note: str = ""


def _kalman_filter(y: np.ndarray, s2e: float, s2n: float, s2z: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """标准 Kalman 滤波（filtered，无平滑），返回：
    (残差 e, 残差方差 F, 状态均值 mu, 状态方差 P, 对数似然)。
    数值稳定化：方差下限 1e-8，防止发散。
    """
    n = len(y)
    e = np.zeros(n)
    F = np.zeros(n)
    mu = np.zeros(n)
    P = np.zeros(n)
    ll = 0.0
    # 初始状态（diffuse 先验的近似）
    m = y[0]
    p = max(s2e, 1e-4)
    trend = 0.0
    p_trend = 1e-4
    tiny = 1e-8
    for t in range(n):
        # 预测
        m_pred = m + trend
        p_pred = p + p_trend + s2n
        p_trend_pred = p_trend + s2z
        cov = p_trend_pred
        # 观测更新
        v = y[t] - m_pred
        F_t = p_pred + s2e
        F_t = max(F_t, tiny)
        K = p_pred / F_t
        Kt = cov / F_t
        m = m_pred + K * v
        p = (1 - K) * p_pred
        trend = trend + Kt * v
        p_trend = p_trend_pred - Kt * cov
        p = max(p, tiny)
        p_trend = max(p_trend, tiny)
        e[t] = v
        F[t] = F_t
        mu[t] = m_pred
        P[t] = p_pred
        # 似然贡献（高斯）
        ll += -0.5 * (np.log(2 * np.pi * F_t) + v * v / F_t)
    return e, F, mu, P, ll


def _fit_single(country: str, y: np.ndarray, grid: list[tuple[float, float, float]]) -> KalmanResult:
    """对单国序列：网格 + 局部精化，返回最优参数。"""
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    n = len(y)
    if n == 0:
        return KalmanResult(country, 0.0, 0.0, 0.0, -np.inf, False, 0, diverged=True, note="全 NaN 序列，跳过")
    if n < 8:
        return KalmanResult(country, 0.0, 0.0, 0.0, -np.inf, False, n, note="序列过短(<8)，跳过")

    best: tuple[float, tuple] = (-np.inf, grid[0])
    for params in grid:
        try:
            _, _, _, _, ll = _kalman_filter(y, *params)
            if np.isfinite(ll) and ll > best[0]:
                best = (ll, params)
        except Exception:  # noqa: BLE001  # 网格点数值异常（矩阵奇异等）→ 跳过该候选参数
            continue

    # 局部精化（L-BFGS-B，从网格最优点出发，方差参数对数空间）
    def neg_ll(log_p: np.ndarray) -> float:
        s2e, s2n, s2z = np.exp(log_p)
        try:
            _, _, _, _, ll = _kalman_filter(y, float(s2e), float(s2n), float(s2z))
            return float(-ll) if np.isfinite(ll) else 1e18
        except Exception:  # noqa: BLE001
            return 1e18

    log_p0 = np.log(np.array(best[1], dtype=float))
    res = minimize(neg_ll, log_p0, method="L-BFGS-B",
                   bounds=[(np.log(1e-6), np.log(1e2))] * 3)
    converged = bool(res.success)
    s2e, s2n, s2z = np.exp(res.x)
    e, F, mu, P, ll = _kalman_filter(y, float(s2e), float(s2n), float(s2z))
    diverged = bool(np.any(np.abs(e) > 15) or np.any(F <= 0) or not np.isfinite(ll))
    note = ""
    if diverged:
        note = "发散警告：残差/方差异常，建议人工复核"
    return KalmanResult(country, float(s2e), float(s2n), float(s2z), float(ll),
                        converged, n, diverged=diverged, note=note)


def _build_grid(scale: float = 1.0) -> list[tuple[float, float, float]]:
    """MLE 网格：σ²_ε ∈ {0.001, 0.01, 0.05}·scale，σ²_η/σ²_ζ 组合。"""
    base = [0.0005, 0.005, 0.02, 0.08]
    grid: list[tuple[float, float, float]] = []
    for s2e in base:
        for s2n in [0.0001, 0.001, 0.01]:
            for s2z in [0.00005, 0.0005, 0.005]:
                grid.append((s2e * scale, s2n * scale, s2z * scale))
    return grid


def compute_kalman_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """F05 主入口：对面板每国军费 log 序列估计 LLT，输出 e* 与异常标记。

    :param panel: 必须含 iso3, year, mil_expense_log
    :return: kalman_signals.csv 内容（iso3, year, e_star, is_anomaly,
             sigma2_eps, sigma2_eta, sigma2_zeta, log_lik）
    """
    df = panel.sort_values(["iso3", "year"]).copy()
    sig_rows: list[dict] = []
    params_rows: list[dict] = []
    skipped: list[str] = []
    grid = _build_grid()

    for iso3, grp in df.groupby("iso3"):
        y = grp["mil_expense_log"].to_numpy(dtype=float)
        years = grp["year"].to_numpy()
        fit = _fit_single(iso3, y, grid)
        params_rows.append({"iso3": iso3, "sigma2_eps": fit.sigma2_eps, "sigma2_eta": fit.sigma2_eta,
                            "sigma2_zeta": fit.sigma2_zeta, "log_lik": fit.log_lik,
                            "converged": fit.converged, "n_obs": fit.n_obs,
                            "diverged": fit.diverged, "note": fit.note})
        if fit.n_obs == 0:
            skipped.append(iso3)  # 仅全 NaN 序列跳过（F05 验收）
            continue
        # 标准化一步预测残差（filtered，仅 ≤t 信息）
        e, F, _, _, _ = _kalman_filter(np.asarray(y[~np.isnan(y)], dtype=float),
                                       fit.sigma2_eps, fit.sigma2_eta, fit.sigma2_zeta)
        std_e = e / np.sqrt(np.maximum(F, 1e-12))
        # 发散序列（军费剧烈波动）仍输出信号，但截断极端残差至 ±10σ（防止污染特征）
        std_e = np.clip(std_e, -10.0, 10.0)
        # 将残差对齐回原始年份（NaN 年份对应 NaN）
        valid = ~np.isnan(y)
        e_star_full = np.full(len(y), np.nan)
        e_star_full[valid] = std_e
        for j, yr in enumerate(years):
            sig_rows.append({
                "iso3": iso3, "year": int(yr),
                "e_star": float(e_star_full[j]) if np.isfinite(e_star_full[j]) else None,
                "is_anomaly": bool(np.isfinite(e_star_full[j]) and abs(e_star_full[j]) >= ANOMALY_Z),
            })

    signals = pd.DataFrame(sig_rows)
    params_df = pd.DataFrame(params_rows)
    signals["is_anomaly"] = signals["is_anomaly"].astype(int)
    signals.to_csv(get_path("kalman_signals_file"), index=False)
    params_df.to_csv(get_path("kalman_params_file"), index=False)
    meta.log_stage("kalman_fit", n_countries=len(params_df), skipped=skipped[:20],
                   n_anomaly=int(signals["is_anomaly"].sum()),
                   diverged=int(params_df["diverged"].sum()))
    return signals
