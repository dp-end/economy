"""Yerel model 1.5 (örnek soru S4): calculate_fundamental_change KAP tarihine göre çeyrek seçimi, açıklama sonrası
1/5/20 günlük tepki (XU100'e ve sektöre göre), iyileşen / diğer çeyrek özeti."""
import asyncio
import json

import numpy as np
import pandas as pd
import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

import src.events.earnings_reaction as ER
from src.data.prices import get_index_history, get_stock_history
from src.mcp_server import tools as T
from src.mcp_server.server import build_server
from src.reports.r4_earnings import sector_abnormal

CEIL = "2026-09-10"
Q4 = dict(ticker="TUPRS", start="2024-01-01", end=CEIL)


def call(name, args):
    async def go():
        async with Client(build_server()) as c:
            r = await c.call_tool(name, args)
            return r.is_error, "".join(getattr(x, "text", "") for x in r.content)
    return asyncio.run(go())


def q4():
    with T.data_ceiling(CEIL):
        err, text = call("calculate_fundamental_change", Q4)
    assert not err, text
    return json.loads(text)


def test_quarters_selected_by_kap_reaction_day():
    out = q4()
    ev = pd.read_parquet("reports/earnings_events.parquet")
    ev = ev[(ev["ticker"] == "TUPRS") & ev["event_day"].between("2024-01-01", CEIL)]
    assert [r["period_end"][:10] for r in out["quarters"]] == [str(d.date()) for d in ev["period_end"]]
    assert [r["publish_time"][:16] for r in out["quarters"]] == [str(p)[:16] for p in ev["publish_time"]]
    assert out["asof"] == CEIL and len(out["quarters"]) == 11


def test_reaction_matches_r4_and_direct_prices():
    out = q4()
    ev = pd.read_parquet("reports/earnings_events.parquet")
    ev = ev[(ev["ticker"] == "TUPRS") & ev["event_day"].between("2024-01-01", CEIL)].reset_index(drop=True)
    peers = sector_abnormal(ev.copy())                                    # R4'ün akran sepeti hesabı
    c, xu = get_stock_history("TUPRS")["Close"], get_index_history("XU100")["Close"]
    for i, r in enumerate(out["quarters"]):
        x = r["reaction"]
        assert x["ab_xu100_1_pct"] == pytest.approx(ev.loc[i, "d0"] * 100)          # R4 d0
        assert x["ab_xu100_5_pct"] == pytest.approx(ev.loc[i, "d0_4"] * 100)        # R4 d0_4
        assert x["ab_sector_1_pct"] == pytest.approx(peers.loc[i, "d0_vs_peers"])
        assert x["ab_sector_5_pct"] == pytest.approx(peers.loc[i, "d0_4_vs_peers"])
        pos = c.index.get_loc(pd.Timestamp(r["event_day"][:10]))                   # 20 gün: fiyattan doğrudan
        lo, hi = c.index[pos - 1], c.index[pos + 19]
        assert x["window_end_20"] == str(hi.date()) and x["complete_20"]
        assert x["ret_20_pct"] == pytest.approx((c[hi] / c[lo] - 1) * 100)
        assert x["ab_xu100_20_pct"] == pytest.approx((c[hi] / c[lo] - xu[hi] / xu[lo]) * 100)
    last = out["quarters"][-1]                      # 2026/6: tepki 05.08.2026, 20 gün 01.09.2026'da kapandı
    assert last["period_end"].startswith("2026-06-30") and last["reaction"]["window_end_20"] == "2026-09-01"


def test_improving_flags_and_group_summary():
    out = q4()
    for r in out["quarters"]:
        mu, nd = r["ebitda_margin_chg"], r["deleveraging"]
        assert r["improving"] == (None if mu is None or nd is None else (mu > 0 and nd > 0))
    for g in out["reaction_summary"]:
        rows = [r for r in out["quarters"] if r["improving"] is {"iyilesen": True, "diger": False,
                                                                  "bilinmiyor": None}[g["group"]]]
        assert g["n_quarters"] == len(rows)
        for h in g.get("horizons", []):
            vals = [r["reaction"][f"ab_xu100_{h['horizon_days']}_pct"] for r in rows]
            assert h["n"] == len(vals) and h["mean_ab_xu100_pct"] == pytest.approx(np.mean(vals))
    assert any("nedensellik" in w for w in out["warnings"]) and any("Az çeyrek" in w for w in out["warnings"])


def test_open_window_left_out_until_it_closes():
    with T.data_ceiling("2026-08-20"):                  # 2026/6 tepki günü 05.08: 5 gün kapandı, 20 gün henüz değil
        out = T.calculate_fundamental_change(ticker="TUPRS", start="2026-01-01")
    last = out["quarters"][-1]["reaction"]
    assert last["complete_5"] and not last["complete_20"]
    assert last["ret_20_pct"] is None and last["ab_xu100_20_pct"] is None and last["window_end_20"] is None
    assert any("2026-06-30" in w and "kapanmamış" in w for w in out["warnings"])
    imp = next(g for g in out["reaction_summary"] if g["group"] == "iyilesen")
    h20 = next(h for h in imp["horizons"] if h["horizon_days"] == 20)
    assert h20["n"] == imp["n_quarters"] - 1                                              # özete katılmadı


def test_reaction_ignores_prices_after_ceiling(monkeypatch):
    a = q4()
    idx, px, xu, groups = ER._reaction_panel()
    cut = idx[idx <= CEIL]
    monkeypatch.setattr(ER, "_reaction_panel", lambda: (cut, px.loc[cut], xu.loc[cut], groups))   # sonrası yok
    b = T.calculate_fundamental_change(**Q4)
    assert [r["reaction"] for r in a["quarters"]] == [r["reaction"] for r in b["quarters"]]
    assert a["reaction_summary"] == b["reaction_summary"]


def test_bank_and_errors():
    with T.data_ceiling(CEIL):
        out = T.calculate_fundamental_change(ticker="AKBNK", start="2025-01-01")
    assert all(r["improving"] is None for r in out["quarters"])
    assert any("FAVÖK ve net borç hesaplanmaz" in w for w in out["warnings"])
    with pytest.raises(ToolError, match="sonra olamaz"):
        T.calculate_fundamental_change(ticker="TUPRS", start="2026-05-01", end="2026-01-01")


def test_last_quarter_disclosure_age_in_trading_days():
    """v3 (26 Kasım 2026): kural 'son 63 işlem gününde açıklandı' için tepki gününden bu yana geçen işlem günü
    araçta. TUPRS 2026/06: tepki günü 2026-08-05, 10.09.2026'ya kadar 26 işlem günü."""
    o = T.calculate_fundamental_change(**Q4)
    lq = o["last_quarter"]
    assert lq["period_end"] == o["quarters"][-1]["period_end"] and lq["improving"] == o["quarters"][-1]["improving"]
    days = get_index_history("XU100").index
    ev = pd.Timestamp(lq["event_day"])
    assert lq["event_day"] == "2026-08-05" and lq["trading_days_since_disclosure"] == 26
    assert lq["trading_days_since_disclosure"] == int(((days > ev) & (days <= pd.Timestamp(CEIL))).sum())
    same_day = T.calculate_fundamental_change(**dict(Q4, end=lq["event_day"]))["last_quarter"]
    assert same_day["event_day"] == lq["event_day"] and same_day["trading_days_since_disclosure"] == 0
