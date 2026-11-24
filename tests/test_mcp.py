import asyncio
import json
import re

import pandas as pd
from mcp import Client

from src.data.prices import get_index_history, get_stock_history
from src.harness.evidence import run_analysis
from src.mcp_server.client_demo import DEMO_CALLS
from src.mcp_server.server import build_server
from src.mcp_server.tools import CEILING_ENV, data_ceiling

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
    assert out["valuation"]["date"][:10] <= "2026-10-27" and out["valuation"]["pe"] > 0  # o günün değerlemesi
    err, text = call("get_news_events", dict(ticker="TURSG", start="2026-10-01", asof="2026-10-20"))
    assert all(e["event_day"][:10] <= "2026-10-20" for e in json.loads(text)["events"])


def test_reversal_event_set_filter():
    """Adım 3: harness 'harness' kümesini ister; varsayılan tüm 22 türü ve olay anındaki durumları döndürür."""
    from src.events.technical_events import HARNESS_EVENTS
    args = dict(ticker="ASELS", start="2025-01-01", end="2026-06-30")
    _, all_txt = call("detect_reversal_events", args)
    _, h_txt = call("detect_reversal_events", dict(args, event_set="harness"))
    a, h = json.loads(all_txt), json.loads(h_txt)
    assert {e["event"] for e in h["events"]} <= set(HARNESS_EVENTS) and h["n_events"] < a["n_events"]
    assert "rsi_state" in a["events"][0] and "k1_confirmed" in a["events"][0]
    err, _ = call("detect_reversal_events", dict(args, event_set="hepsi"))
    assert err


# ---------- tarih tavanı: yerel model soruları "analiz tarihi 10.09.2026" der, sonrası hiç görünmemeli ----------
CEIL, LATER = "2026-09-10", "2026-11-06"
CEILING_CALLS = {        # her araç: (varsayılanlarla, tavandan sonrasını isteyerek)
    "get_stock_history": [dict(ticker="ASELS", start="2026-08-01"), dict(ticker="ASELS", start="2026-08-01", end=LATER)],
    "get_index_history": [dict(start="2026-08-01"), dict(start="2026-08-01", end=LATER)],
    "calculate_indicators": [dict(ticker="ASELS", start="2026-08-01"),
                             dict(ticker="ASELS", start="2026-08-01", end=LATER)],
    "detect_reversal_events": [dict(ticker="ASELS", start="2026-06-01"),
                               dict(ticker="ASELS", start="2026-06-01", end=LATER)],
    "find_support_resistance": [dict(ticker="ASELS"), dict(ticker="ASELS", date=LATER)],
    "rank_sector_relative_strength": [dict(sector="Retail & Consumer", horizon=20),
                                      dict(sector="Retail & Consumer", horizon=20, date=LATER)],
    "test_weekday_pattern": [dict(ticker="THYAO", start="2026-01-01"), dict(ticker="THYAO", start="2026-01-01", end=LATER)],
    "get_fundamental_metrics": [dict(ticker="TUPRS"), dict(ticker="TUPRS", asof=LATER)],
    "calculate_fundamental_change": [dict(ticker="TUPRS", periods=4), dict(ticker="TUPRS", periods=4, asof=LATER)],
    "get_macro_context": [dict(start="2026-08-01"), dict(start="2026-08-01", end=LATER)],
    "get_news_events": [dict(ticker="EREGL", start="2026-08-01"),
                        dict(ticker="EREGL", start="2026-08-01", end=LATER, asof=LATER)],
    "get_speech_features": [dict()],                          # kaynak kimliğiyle istek ayrı testte
    "run_backtest": [dict(strategy="B_sektor", start="2026-01-01"),
                     dict(strategy="B_sektor", start="2026-01-01", end=LATER)],
    "check_data_quality": [dict(ticker="EREGL"), dict(ticker="EREGL", asof=LATER)],
    "build_evidence_bundle": [dict(ticker="EREGL"), dict(ticker="EREGL", asof=LATER)],
}


def test_ceiling_covers_every_tool_and_hides_later_dates():
    assert set(CEILING_CALLS) == SPEC_TOOLS               # yeni araç eklenirse bu teste de girmek zorunda

    async def go(c):
        out = {}
        for name, arglist in CEILING_CALLS.items():
            for i, args in enumerate(arglist):
                r = await c.call_tool(name, args)
                out[name, i] = (r.is_error, "".join(getattr(x, "text", "") for x in r.content))
        return out
    with data_ceiling(CEIL):
        res = run(_session(go))
    for (name, i), (err, text) in res.items():
        assert not err, (name, text[:200])
        out = json.loads(text)
        assert out["data_ceiling"] == CEIL, name
        assert max(re.findall(r"\d{4}-\d{2}-\d{2}", text)) <= CEIL, name      # tavandan sonraki hiçbir tarih yok
        assert bool(out["date_warnings"]) == (i == 1), (name, out["date_warnings"])  # kırpıldıysa söylendi


def test_ceiling_clamp_equals_explicit_end_and_masks_forward_returns():
    with data_ceiling(CEIL):
        _, clamped = call("get_stock_history", dict(ticker="ASELS", start="2026-08-01", end=LATER))
        err, text = call("get_stock_history", dict(ticker="ASELS", start="2026-10-01"))
        assert err and "analiz tarihinden" in text                             # tümüyle tavandan sonraki dönem
    _, explicit = call("get_stock_history", dict(ticker="ASELS", start="2026-08-01", end=CEIL))
    assert json.loads(clamped)["rows"] == json.loads(explicit)["rows"]
    # Tavana 10 işlem gününden yakın olayların 10 günlük penceresi henüz kapanmadı: tavan varken boş, yokken dolu
    days = get_index_history("XU100").index
    first = days[days.get_loc(pd.Timestamp(CEIL)) - 9]
    te = pd.read_parquet("reports/technical_events.parquet")
    ticker = te[te["date"].between(first, CEIL)]["ticker"].iloc[0]
    args = dict(ticker=ticker, start=str(first.date()))
    with data_ceiling(CEIL):
        _, a = call("detect_reversal_events", args)
    _, b = call("detect_reversal_events", args)
    ea = json.loads(a)["events"]
    eb = [e for e in json.loads(b)["events"] if e["date"] <= CEIL]
    assert ea and len(ea) == len(eb)
    assert all(e["exc_10"] is None for e in ea) and all(e["exc_10"] is not None for e in eb)


def test_ceiling_hides_later_speech_sources_and_log_records():
    def ids(text):
        return [s["source_id"] for s in json.loads(text)["sources"]]
    with data_ceiling("2026-10-28"):            # Fed 28.10.2026 21:30 (kapanış sonrası); 29 Ekim tatil -> 30 Ekim
        _, text = call("get_speech_features", {})
        assert "fed_20261028" not in ids(text)
        err, text = call("get_speech_features", dict(source_id="fed_20261028"))
        assert err and "bilinmeyen" in text
    with data_ceiling("2026-10-30"):
        _, text = call("get_speech_features", {})
        assert ids(text) == ["fed_20261028"]                                    # TCMB 12.11.2026 henüz yok
    with data_ceiling(CEIL):
        err, text = call("build_evidence_bundle", dict(analysis_id="2f8d0f98ef03"))      # kayıt asof 2026-11-06
        assert err and "analiz tarihinden sonraki" in text
        err, text = call("build_evidence_bundle", dict(analysis_id="9b89c803932e"))      # kayıt asof 2026-01-09
        assert not err and json.loads(text)["asof"].startswith("2026-01-09")


def test_ceiling_from_env_var_and_unchanged_without_ceiling(monkeypatch):
    monkeypatch.setenv(CEILING_ENV, CEIL)                  # ayrı süreçteki MCP sunucusu bu yolla ayarlanır
    out = json.loads(call("get_index_history", dict(start="2026-09-01"))[1])
    assert out["data_ceiling"] == CEIL and out["rows"][-1]["date"] == CEIL
    monkeypatch.delenv(CEILING_ENV)
    out = json.loads(call("get_index_history", dict(start="2026-09-01"))[1])
    assert "data_ceiling" not in out and out["rows"][-1]["date"] == LATER    # tavan yoksa eski davranış
