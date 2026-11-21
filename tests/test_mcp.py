import asyncio
import json

import pandas as pd
from mcp import Client

from src.data.prices import get_stock_history
from src.harness.evidence import run_analysis
from src.mcp_server.client_demo import DEMO_CALLS
from src.mcp_server.server import build_server

SPEC_TOOLS = {"get_stock_history", "get_index_history", "calculate_indicators", "detect_reversal_events",
              "find_support_resistance", "rank_sector_relative_strength", "test_weekday_pattern",
              "get_fundamental_metrics", "calculate_fundamental_change", "get_macro_context", "get_news_events",
              "get_speech_features", "run_backtest", "check_data_quality", "build_evidence_bundle"}


def run(coro):
    return asyncio.run(coro)


async def _session(fn):
    async with Client(build_server()) as c:
        return await fn(c)


def call(name, args):
    async def go(c):
        r = await c.call_tool(name, args)
        text = "".join(getattr(x, "text", "") for x in r.content)
        return r.is_error, text
    return run(_session(go))


def test_lists_exactly_the_15_spec_tools():
    tools = run(_session(lambda c: c.list_tools())).tools
    assert {t.name for t in tools} == SPEC_TOOLS
    assert all(t.description for t in tools)


def test_every_tool_callable_and_json():
    async def go(c):
        out = {}
        for name, args in DEMO_CALLS.items():
            r = await c.call_tool(name, args)
            out[name] = (r.is_error, "".join(getattr(x, "text", "") for x in r.content))
        return out
    for name, (err, text) in run(_session(go)).items():
        assert not err, (name, text[:200])
        json.loads(text)


def test_invalid_input_returns_error_not_crash():
    for name, args, word in [("get_stock_history", dict(ticker="YOKBOYLE"), "bilinmeyen hisse"),
                             ("run_backtest", dict(strategy="AL_SAT"), "bilinmeyen strateji"),
                             ("get_fundamental_metrics", dict(ticker="ASELS", asof="2030-01-01"), "gelecek verisi yok"),
                             ("get_stock_history", dict(ticker="ASELS", start="dün"), "tarih değil")]:
        err, text = call(name, args)
        assert err and word in text, (name, text)


def test_tool_outputs_match_module_functions():
    err, text = call("get_stock_history", dict(ticker="ASELS", start="2026-11-02", end="2026-11-06"))
    rows = json.loads(text)["rows"]
    direct = get_stock_history("ASELS").loc["2026-11-02":"2026-11-06", "Close"]
    assert [r["Close"] for r in rows] == direct.tolist()
    err, text = call("build_evidence_bundle", dict(ticker="TURSG", asof="2026-11-06"))
    b, r = json.loads(text), run_analysis("TURSG", "2026-11-06")
    assert b["action"] == r["action"] and b["gate"]["status"] == r["gate"]["status"]
    assert [e["text"] for e in b["committed"]] == [e["text"] for e in r["committed"]]


def test_tools_do_not_leak_future():
    err, text = call("detect_reversal_events", dict(ticker="PETKM", start="2026-10-01", end="2026-10-30"))
    ev = pd.DataFrame(json.loads(text)["events"])
    assert (pd.to_datetime(ev["date"]) <= "2026-10-30").all()
    late = ev[pd.to_datetime(ev["date"]) > pd.Timestamp("2026-10-23")]          # 5 günlük pencere 30 Ekim'i geçer
    assert late["exc_5"].isna().all()
    err, text = call("get_fundamental_metrics", dict(ticker="ASELS", asof="2026-10-27"))
    out = json.loads(text)
    assert out["period_end"].startswith("2026-06-30")                            # 2026/9 henüz açıklanmadı
    assert out["unit_flow"] == "TL@2026-06" and out["unit_stock"] == "TL@2026-06"  # o gün görünen son raporun parası
    err, text = call("get_news_events", dict(ticker="TURSG", start="2026-10-01", asof="2026-10-20"))
    assert all(e["event_day"][:10] <= "2026-10-20" for e in json.loads(text)["events"])
