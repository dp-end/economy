"""Yerel model 1.6 (örnek soru S5): get_macro_context dönem özeti, uzun aralıkta ay sonu satırları ve ticker
verilince kur-hisse ilişkisi (günlük beta + kurun hızlı arttığı aylar)."""
import asyncio
import json

import numpy as np
import pandas as pd
import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from src.data.macro import get_macro_context as macro_panel
from src.data.prices import get_index_history, get_stock_history
from src.mcp_server import tools as T
from src.mcp_server.server import build_server

CEIL = "2026-09-10"
Q5 = dict(start="2024-01-01", end=CEIL, ticker="EREGL")


def call(name, args):
    async def go():
        async with Client(build_server()) as c:
            r = await c.call_tool(name, args)
            return r.is_error, "".join(getattr(x, "text", "") for x in r.content)
    return asyncio.run(go())


def q5():
    with T.data_ceiling(CEIL):
        err, text = call("get_macro_context", Q5)
    assert not err, text
    return json.loads(text)


def test_summary_and_decisions_match_panel_and_official_calendar():
    out, m = q5(), macro_panel("2024-01-01", CEIL)
    sm = out["summary"]
    assert sm["start"] == "2024-01-02" and sm["end"] == CEIL
    assert sm["usdtry"]["start"] == m["usdtry"].iloc[0] and sm["usdtry"]["end"] == m["usdtry"].iloc[-1]
    assert sm["usdtry"]["change_pct"] == pytest.approx((m["usdtry"].iloc[-1] / m["usdtry"].iloc[0] - 1) * 100)
    assert sm["policy_rate"]["start"] == 42.5 and sm["policy_rate"]["end"] == 37.0
    cal = pd.read_csv("config/policy_meetings.csv", parse_dates=["date"])
    tcmb = set(cal.loc[cal["bank"] == "TCMB", "date"].dt.strftime("%Y-%m-%d"))
    fed = set(cal.loc[cal["bank"] == "FED", "date"])
    dec = sm["policy_rate"]["decisions"]
    assert len(dec) == 11 and all(d["date"] in tcmb for d in dec)                 # her karar resmi toplantı günü
    assert sum(d["change_pt"] for d in dec) == pytest.approx(37.0 - 42.5)
    assert all(pd.Timestamp(c["date"]) - pd.Timedelta(days=1) in fed for c in sm["fed_range"]["changes"])  # ertesi gün
    assert sm["cpi_yoy"]["end_month"] == "2026-08"                     # Ağustos TÜFE'si 3 Eylül'de yayımlandı


def test_long_range_returns_month_end_rows_short_range_daily():
    out = q5()
    m = macro_panel("2024-01-01", CEIL)
    month_end = m.groupby(m.index.to_period("M")).tail(1)
    assert out["meta"]["frequency"].startswith("ay sonu") and out["n_rows"] == len(month_end) == 33
    assert [r["date"] for r in out["rows"]] == [d.strftime("%Y-%m-%d") for d in month_end.index]
    assert not out["truncated_warning"]
    short = T.get_macro_context(start="2026-08-01", end=CEIL)
    assert short["meta"]["frequency"] == "günlük" and short["n_rows"] == len(macro_panel("2026-08-01", CEIL))


def test_fx_relation_matches_independent_calculation():
    fx = q5()["fx_relation"]
    c = get_stock_history("EREGL")["Close"].loc["2024-01-01":CEIL]
    xu = get_index_history("XU100")["Close"].loc["2024-01-01":CEIL]
    u = macro_panel("2024-01-01", CEIL)["usdtry"]
    d = pd.DataFrame({"r": c.pct_change(), "x": xu.pct_change(), "u": u.pct_change()}).dropna() * 100
    beta = np.polyfit(d["u"], d["r"] - d["x"], 1)[0]                              # OLS eğimi (nokta tahmini)
    assert fx["daily"]["excess_vs_xu100"]["beta"] == pytest.approx(beta) and fx["daily"]["n_days"] == len(d)
    assert fx["daily"]["excess_vs_xu100"]["p"] > 0.05                             # anlamlı ilişki yok
    me = lambda s: s.groupby(s.index.to_period("M")).last()
    mc = lambda s: (me(s) / me(s).shift(1).fillna(s.iloc[0]) - 1) * 100        # ilk ay dönemin ilk gününe göre
    mm = pd.DataFrame({"r": mc(c), "x": mc(xu), "u": mc(u)})
    fast = mm["u"] > mm["u"].median()
    mo = fx["monthly"]
    assert mo["n_months"] == 33 and mo["usdtry_months_up"] == int((mm["u"] > 0).sum()) == 32
    assert mo["fast_fx_months"]["n_months"] == int(fast.sum())
    assert mo["fast_fx_months"]["mean_excess_vs_xu100_pct"] == pytest.approx((mm["r"] - mm["x"])[fast].mean())
    assert mo["other_months"]["mean_ret_pct"] == pytest.approx(mm["r"][~fast].mean())
    assert mo["corr_ret_vs_usdtry"] == pytest.approx(mm["r"].corr(mm["u"]))


def test_macro_ignores_data_after_ceiling(monkeypatch):
    with T.data_ceiling(CEIL):
        a = T.get_macro_context(**Q5)
    st, ix = T._stock, T._index
    monkeypatch.setattr(T, "_stock", lambda t: st(t).loc[:CEIL])                  # 10.09.2026 sonrası fiyat yok
    monkeypatch.setattr(T, "_index", lambda i: ix(i).loc[:CEIL])
    b = T.get_macro_context(**Q5)
    assert {k: v for k, v in a.items() if k not in ("data_ceiling", "date_warnings")} == b


def test_macro_errors():
    with pytest.raises(ToolError, match="bilinmeyen hisse"):
        T.get_macro_context(start="2024-01-01", ticker="YOKBOYLE")
    out = T.get_macro_context(start="2026-11-04", end="2026-11-06")                    # ticker yok: ilişki bloğu yok
    assert "fx_relation" not in out and out["summary"]["start"] == "2026-11-04"
