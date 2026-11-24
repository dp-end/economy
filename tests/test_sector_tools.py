"""Yerel model 1.2 (örnek soru S1): sektör aracına hisse listesi + Türkçe sektör adı, run_backtest catch_up geçmişi."""
import asyncio
import json

import numpy as np
import pandas as pd
import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from src.data.prices import get_stock_history
from src.mcp_server import tools as T
from src.mcp_server.server import build_server

CEIL = "2026-09-10"
GROUP = ["BIMAS", "MGROS", "MAVI"]


def call(name, args):
    async def go():
        async with Client(build_server()) as c:
            r = await c.call_tool(name, args)
            return r.is_error, "".join(getattr(x, "text", "") for x in r.content)
    return asyncio.run(go())


def data_ceiling():
    return T.data_ceiling(CEIL)


def closes(upto=CEIL):
    return pd.DataFrame({t: get_stock_history(t)["Close"] for t in GROUP}).loc[:upto]


def test_rank_with_ticker_list_matches_direct_calculation():
    with data_ceiling():
        err, text = call("rank_sector_relative_strength", dict(tickers=GROUP, horizon=20))   # MCP üzerinden liste
    assert not err, text
    out = json.loads(text)
    p = closes()
    r = p.iloc[-1] / p.iloc[-21] - 1
    assert out["date"] == CEIL and out["most_behind"] == r.idxmin() == "MAVI"
    for m in out["members"]:
        t = m["ticker"]
        assert m["lag"] == pytest.approx(r[t] - r.drop(t).median(), abs=1e-12)            # kendisi hariç
        assert m["gap_to_sector_median"] == pytest.approx(r[t] - r.median(), abs=1e-12)  # kendisi dahil


@pytest.mark.parametrize("name, group", [("Perakende", "Retail & Consumer"), ("BANKACILIK", "Banking"),
                                         ("sağlık", "Healthcare"), ("saglik", "Healthcare"),
                                         ("Demir Çelik", "Metals & Mining"), ("İnşaat", "Construction RE & Utilities"),
                                         ("Banking", "Banking"), ("Teknoloji, telekom ve savunma", "Tech Telecom & Defense"),
                                         ("Automotive & Durables", "Automotive & Durables")])
def test_sector_names_in_turkish(name, group):
    assert T.rank_sector_relative_strength(sector=name, date=CEIL)["sector"] == group


def test_sector_errors():
    with pytest.raises(ToolError, match="bilinmeyen sektör"):
        T.rank_sector_relative_strength(sector="uzay")
    with pytest.raises(ToolError, match="en az 2"):
        T.rank_sector_relative_strength(tickers=["MAVI", "mavi"])
    with pytest.raises(ToolError, match="sector ya da tickers"):
        T.rank_sector_relative_strength()


def _recount(p, focus, thr, start, horizons=(5, 10, 20), gap=10):
    """Araçtan bağımsız sayım: 20 günlük fark ilk kez <= -thr olan günler (en az 10 gün arayla) ve h gün sonrası."""
    r = p.pct_change(20)
    lag = r[focus] - r.drop(columns=focus).median(axis=1)
    events, last = [], -10**9
    for i in range(len(p)):
        now, before = lag.iloc[i] <= -thr, i > 0 and lag.iloc[i - 1] <= -thr
        if now and not before and i - last > gap:
            last = i
            if p.index[i] >= pd.Timestamp(start):
                events.append(i)
    out = {}
    for h in horizons:
        out[h] = []
        for i in events:
            if i + h < len(p):
                g = p.iloc[i + h] / p.iloc[i] - 1
                out[h].append((p.index[i], g[focus] - g.drop(focus).median(), -lag.iloc[i]))
    return events, out


def test_catch_up_matches_independent_recount():
    with data_ceiling():
        b = T.run_backtest(strategy="catch_up", tickers=GROUP, start="2024-01-01")
        b10 = T.run_backtest(strategy="catch_up", tickers=GROUP, start="2024-01-01", threshold=0.10)
    p = closes()
    assert b["ticker"] == "MAVI" and b["end"] == CEIL and b["threshold_source"] == "hissenin bugünkü farkı"
    assert b["threshold"] == pytest.approx(-b["current_lag"])
    events, rel = _recount(p, "MAVI", b["threshold"], "2024-01-01")
    st = b["stock"]
    assert st["n_events"] == len(events) and [r["horizon_days"] for r in st["by_horizon"]] == [5, 10, 20]
    for row in st["by_horizon"]:
        x = rel[row["horizon_days"]]
        a = np.array([v for _, v, _ in x])
        assert row["n"] == len(x) and row["n_incomplete"] == len(events) - len(x)
        assert row["hit_rate"] == pytest.approx((a > 0).mean())
        assert row["full_close_rate"] == pytest.approx(np.mean([v >= need for _, v, need in x]))
        assert row["median_rel"] == pytest.approx(np.median(a))
        assert row["mean_rel_after_cost"] == pytest.approx(a.mean() - 2 * b["cost_one_way"])
    fp = [d.strftime("%Y-%m-%d") for d, v, _ in rel[20] if v <= 0]
    assert [f["date"] for f in st["false_positives"]] == fp
    assert st["false_positive_rate_20"] == pytest.approx(len(fp) / len(rel[20])) and st["n_false_positives"] == len(fp)
    # Eşik istekte verilirse o kullanılır (R2 kuralı: 20 günde 10 puan)
    assert b10["threshold_source"] == "istekte verilen"
    assert b10["stock"]["n_events"] == len(_recount(p, "MAVI", 0.10, "2024-01-01")[0])


def test_catch_up_ignores_data_after_ceiling(monkeypatch):
    with data_ceiling():
        a = T.run_backtest(strategy="catch_up", tickers=GROUP, start="2024-01-01")
    full = T._stock
    monkeypatch.setattr(T, "_stock", lambda t: full(t).loc[:CEIL])           # 10.09.2026 sonrası fiyat hiç yok
    b = T.run_backtest(strategy="catch_up", tickers=GROUP, start="2024-01-01", end=CEIL)
    assert {k: v for k, v in a.items() if k not in ("data_ceiling", "date_warnings")} == b
    assert all(r["n_incomplete"] >= 1 for r in a["stock"]["by_horizon"])     # son olayın penceresi açık, sayılmadı


def test_catch_up_errors():
    with pytest.raises(ToolError, match="yalnızca strategy='catch_up'"):
        T.run_backtest(strategy="A_teknik", tickers=GROUP)
    with data_ceiling():
        with pytest.raises(ToolError, match="gerisinde değil"):
            T.run_backtest(strategy="catch_up", tickers=GROUP, ticker="BIMAS")
        with pytest.raises(ToolError, match="lookback"):
            T.run_backtest(strategy="catch_up", tickers=GROUP, lookback=7)
        with pytest.raises(ToolError, match="grupta yok"):
            T.run_backtest(strategy="catch_up", tickers=GROUP, ticker="ASELS")
    err, text = call("run_backtest", dict(strategy="catch_up", sector="perakende", start="2024-01-01", end=CEIL))
    assert not err and json.loads(text)["sector"] == "Retail & Consumer"                  # MCP + Türkçe ad
