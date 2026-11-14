"""F01 军费数据自动采集：World Bank WDI API 抓取 + 缓存 + 降级。

- 从 WDI API 抓取军费支出（MS.MIL.XPND.CD）与经济协变量（GDP 增速/通胀/失业率）
- 缓存至 data/raw_real.json（PRD F01：支持 160+ 国 1949-至今）
- 异常路径（PRD §5.3）：API 超时/空响应 → 使用缓存或模拟数据 + 标记数据时间
"""
from __future__ import annotations

import json
import time
from typing import Any, Optional

import pandas as pd
import requests

from . import meta
from .paths import cfg, get_path


class DataSourceUnavailable(Exception):
    """数据源不可达（PRD §5.3 异常路径：数据源不可达）。"""


def _indicator_map() -> dict[str, str]:
    ind = cfg()["data"]["wdi"]["indicators"]
    return {"mil_expense": ind["mil_expense"], "gdp_growth": ind["gdp_growth"],
            "inflation": ind["inflation"], "unemployment": ind["unemployment"]}


def fetch_wdi_live(start_year: int, end_year: int, timeout_sec: Optional[int] = None,
                   retries: Optional[int] = None) -> pd.DataFrame:
    """真实调用 World Bank WDI API，返回长表 DataFrame（iso3, year, indicator, value）。

    :raises DataSourceUnavailable: API 不可达 / 响应异常 / 结果为空。
    """
    wdi_cfg = cfg()["data"]["wdi"]
    timeout_sec = timeout_sec or wdi_cfg["timeout_sec"]
    retries = retries or wdi_cfg["retries"]
    fmt = wdi_cfg["format"]
    url_tpl = wdi_cfg["base_url"] + "{ind}" + fmt
    frames: list[pd.DataFrame] = []
    for name, code in _indicator_map().items():
        url = url_tpl.format(ind=code, start=start_year, end=end_year)
        resp: Optional[requests.Response] = None
        last_err: Optional[Exception] = None
        for attempt in range(retries + 1):
            try:
                resp = requests.get(url, timeout=timeout_sec)
                if resp.status_code == 200:
                    break
                last_err = DataSourceUnavailable(f"HTTP {resp.status_code}")
            except Exception as e:  # noqa: BLE001
                last_err = e
            if attempt < retries:
                time.sleep(1.5 * (attempt + 1))
        if resp is None or resp.status_code != 200:
            raise DataSourceUnavailable(f"WDI 指标 {name} 抓取失败: {last_err}")
        data = resp.json()
        # WDI 返回 [pageMeta, [records]]
        records = data[1] if isinstance(data, list) and len(data) > 1 else data
        rows = []
        for r in records:
            iso3 = (r.get("countryiso3code") or "").strip()
            value = r.get("value")
            year = r.get("date")
            if iso3 and value is not None and year and year.isdigit():
                rows.append({"iso3": iso3, "year": int(year), "indicator": name, "value": float(value)})
        if not rows:
            raise DataSourceUnavailable(f"WDI 指标 {name} 返回空记录")
        frames.append(pd.DataFrame(rows))
        meta.log_stage("wdi_fetch", indicator=name, n_rows=len(rows), status="ok")
    df = pd.concat(frames, ignore_index=True)
    df = df.pivot_table(index=["iso3", "year"], columns="indicator", values="value").reset_index()
    df.columns.name = None
    return df


def load_simulated_wdi() -> pd.DataFrame:
    """读取模拟 WDI 数据（降级运行）。"""
    sim_path = get_path("simulated_data_file")
    if not sim_path.exists():
        from . import simulate

        simulate.write_simulated_data()
    with open(sim_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    df = pd.DataFrame(payload["records"])
    return df


def get_military_and_economic_data(start_year: Optional[int] = None,
                                   end_year: Optional[int] = None,
                                   force_refresh: bool = False) -> dict[str, Any]:
    """F01 统一入口：返回 {df, source, fetched_at, cache_used}。

    优先级：真实 WDI 抓取 → data/raw_real.json 缓存 → 模拟数据（标记 simulated）。
    返回结构同时被 F03 清洗与 F04 面板构建消费。
    """
    cfg_data = cfg()["data"]
    start_year = start_year or cfg_data["panel"]["start_year"]
    end_year = end_year or cfg_data["panel"]["end_year"]
    cache_path = get_path("raw_real_file")
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if not force_refresh and cache_path.exists():
        with open(cache_path, "r", encoding="utf-8") as f:
            cached = json.load(f)
        meta.log_stage("wdi_cache_hit", path=str(cache_path))
        return {"df": pd.DataFrame(cached["records"]), "source": cached.get("source", "cache"),
                "fetched_at": cached.get("fetched_at"), "cache_used": True}

    try:
        df = fetch_wdi_live(start_year, end_year)
        source = "wdi-live"
        fetched_at = meta.utcnow_iso()
        meta.log_stage("wdi_live_success", n_countries=df["iso3"].nunique(), years=f"{df['year'].min()}-{df['year'].max()}")
    except DataSourceUnavailable as e:
        meta.log_stage("wdi_live_failed", error=str(e))
        if cfg_data["wdi"].get("simulated_fallback", True):
            df = load_simulated_wdi()
            source = "simulated"
            fetched_at = meta.utcnow_iso()
            meta.log_stage("wdi_fallback_simulated", error=str(e))
        else:
            raise DataSourceUnavailable("WDI 不可达且未启用模拟降级") from e

    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump({"meta": {"source": source, "fetched_at": fetched_at,
                            "start_year": start_year, "end_year": end_year},
                   "records": df.to_dict(orient="records")}, f, ensure_ascii=False, indent=1)
    meta.log_stage("wdi_cache_write", path=str(cache_path), source=source)
    return {"df": df, "source": source, "fetched_at": fetched_at, "cache_used": False}
