"""Yerel model 1.4 (örnek soru S3): detect_reversal_events bant dokunuşu (bb_lower_touch / bb_upper_touch) ve
koşullara göre özet (by_condition). R1'in olay tablosu değişmez."""
import asyncio
import json

import numpy as np
import pandas as pd
import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

import src.events.technical_events as TE
from src.data.prices import get_index_history, get_stock_history
from src.mcp_server import tools as T
from src.mcp_server.server import build_server

CEIL = "2026-09-10"
Q3 = dict(ticker="ASELS", event="bb_lower_touch", start="2024-01-01", end=CEIL)


def call(name, args):
    async def go():
        async with Client(build_server()) as c:
            r = await c.call_tool(name, args)
            return r.is_error, "".join(getattr(x, "text", "") for x in r.content)
    return asyncio.run(go())


def group(out, name):
    return next(g for g in out["by_condition"] if g["condition"] == name)


def horizon(g, h):
    return next(x for x in g["horizons"] if x["horizon_days"] == h)


def test_q3_touches_match_independent_recount():
    """Bant fiyattan bağımsız hesaplanır (SMA20 ± 2 std, ddof=0); dokunuş = Low ≤ alt bant; seri başı, 5 gün arayla."""
    with T.data_ceiling(CEIL):
        err, text = call("detect_reversal_events", Q3)
    assert not err, text
    out = json.loads(text)
    px = get_stock_history("ASELS")
    c = px["Close"]
    lower = c.rolling(20).mean() - 2 * c.rolling(20).std(ddof=0)
    touch = (px["Low"] <= lower).to_numpy()
    days = c.index[c.index <= CEIL]
    events, last = [], -10**9
    for i in range(len(days)):
        if touch[i] and not (i > 0 and touch[i - 1]) and i - last > TE.DEDUPE_GAP:
            last = i
            if days[i] >= pd.Timestamp("2024-01-01"):
                events.append(i)
    assert out["n_events"] == len(events) == group(out, "hepsi")["n_events"]
    assert [e["date"] for e in out["events"]] == [days[i].strftime("%Y-%m-%d") for i in events]
    assert out["meta"]["touch_days"] == int(touch[(c.index >= "2024-01-01") & (c.index <= CEIL)].sum())
    for h in (1, 3, 5, 10):
        r = np.array([c.iloc[i + h] / c.iloc[i] - 1 for i in events if i + h < len(days)])
        x = horizon(group(out, "hepsi"), h)
        assert x["n"] == len(r) and x["success"] == int((r > 0).sum()) and x["failure"] == int((r <= 0).sum())
        assert x["mean_pct"] == pytest.approx(r.mean() * 100) and x["median_pct"] == pytest.approx(np.median(r) * 100)


def test_condition_groups_follow_event_states():
    with T.data_ceiling(CEIL):
        out = T.detect_reversal_events(**Q3, conditions=["rsi_dip+hacim_yuksek", "st_asagi+bulut_alti", "trend_yukari"])
    ev = pd.DataFrame(out["events"])
    assert [g["condition"] for g in out["by_condition"]] == ["hepsi", "rsi_dip+hacim_yuksek", "st_asagi+bulut_alti",
                                                              "trend_yukari"]
    for name, mask in {"rsi_dip+hacim_yuksek": (ev["rsi_state"] == "dip") & ev["vol_recent"],
                       "st_asagi+bulut_alti": (ev["st_state"] == "asagi") & (ev["cloud_pos"] == "altinda"),
                       "trend_yukari": ev["regime"] == "uptrend"}.items():
        g = group(out, name)
        assert g["n_events"] == int(mask.sum())
        assert horizon(g, 5)["success"] == int((ev.loc[mask, "fwd_5"].dropna() > 0).sum())
        assert ("warning" in g) == (horizon(g, 5)["n"] < T.MIN_EVENTS)                  # az olayda uyarı
    with T.data_ceiling(CEIL):
        default = T.detect_reversal_events(**Q3)                                         # varsayılan tekli + ikili set
    assert [g["condition"] for g in default["by_condition"]][1:] == T.DEFAULT_CONDITIONS[1]


def test_recent_touches_left_out_until_their_window_closes():
    """Tavana 3 işlem günü kala olan dokunuş: 1 ve 3 günlük getirisi var, 5 ve 10 günlüğü henüz yok (sayılmaz)."""
    days = get_index_history("XU100").index
    ev = pd.concat(TE.touch_table(t)[0].assign(ticker=t) for t in ("ASELS", "THYAO", "TUPRS", "BIMAS"))
    ev = ev[(ev.index >= "2025-01-01") & (ev["event"] == "bb_lower_touch")]
    d = ev.index[0]
    ceil = days[days.get_loc(d) + 3]
    with T.data_ceiling(ceil):
        out = T.detect_reversal_events(ticker=ev["ticker"].iloc[0], event="bb_lower_touch", start=str(d.date()))
    g = group(out, "hepsi")
    assert out["events"][-1]["date"] <= str(ceil.date())
    assert horizon(g, 3)["n_incomplete"] == 0 and horizon(g, 5)["n_incomplete"] >= 1 and horizon(g, 10)["n_incomplete"] >= 1
    assert all(e["fwd_5"] is None for e in out["events"] if e["date"] == str(d.date()))


def test_touch_result_ignores_data_after_ceiling(monkeypatch):
    with T.data_ceiling(CEIL):
        a = T.detect_reversal_events(**Q3)
    ff, sh = TE.full_features, TE.get_stock_history
    monkeypatch.setattr(TE, "full_features", lambda t: ff(t).loc[:CEIL])                 # 10.09.2026 sonrası yok
    monkeypatch.setattr(TE, "get_stock_history", lambda t: sh(t).loc[:CEIL])
    TE.touch_table.cache_clear()
    try:
        b = T.detect_reversal_events(**Q3)
    finally:
        TE.touch_table.cache_clear()
    assert a["by_condition"] == b["by_condition"] and a["events"] == b["events"]


def test_upper_touch_success_means_price_fell():
    with T.data_ceiling(CEIL):
        out = T.detect_reversal_events(ticker="ASELS", event="bb_upper_touch", start="2024-01-01")
    ev = pd.DataFrame(out["events"])
    g = group(out, "hepsi")
    assert horizon(g, 5)["success"] == int((ev["fwd_5"].dropna() < 0).sum())
    assert [x["condition"] for x in out["by_condition"]][1:] == T.DEFAULT_CONDITIONS[-1]


def test_reversal_tool_errors_and_research_table_untouched():
    with pytest.raises(ToolError, match="bilinmeyen koşul"):
        T.detect_reversal_events(**Q3, conditions=["rsi_dibi"])
    with pytest.raises(ToolError, match="tek bir event"):
        T.detect_reversal_events(ticker="ASELS", conditions=["rsi_dip"])
    with pytest.raises(ToolError, match="bilinmeyen olay"):
        T.detect_reversal_events(ticker="ASELS", event="bb_lower_touch", event_set="harness")
    te = pd.read_parquet("reports/technical_events.parquet")
    assert not te["event"].isin(list(TE.TOUCH_DIRECTION)).any()                          # R1 tablosuna girmedi
