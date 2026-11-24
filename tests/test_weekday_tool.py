"""Yerel model 1.3 (örnek soru S2): test_weekday_pattern 'week:...' dizisi, oos_start ile örneklem içi/dışı ve
kural tabanlı kararlılık kararı."""
import asyncio
import json

import numpy as np
import pandas as pd
import pytest
from mcp import Client
from mcp.server.mcpserver.exceptions import ToolError

from src.backtest.engine import COST_ONE_WAY
from src.data.prices import get_stock_history
from src.mcp_server import tools as T
from src.mcp_server.server import build_server

CEIL = "2026-09-10"
Q2 = dict(ticker="THYAO", pattern="week:DDD-UU", start="2024-01-01", end=CEIL, oos_start="2026-01-01")
LABEL = "koşul: Pzt düşüş, Sal düşüş, Çar düşüş -> sonuç: Per yükseliş, Cum yükseliş"


def call(name, args):
    async def go():
        async with Client(build_server()) as c:
            r = await c.call_tool(name, args)
            return r.is_error, "".join(getattr(x, "text", "") for x in r.content)
    return asyncio.run(go())


def recount(ticker, start, end):
    """Araçtan bağımsız: fiyattan günlük getiri; Pzt-Cum 5 günü olan ve tatil sonrası günü olmayan haftalar
    (Pazartesi önceki işlem gününden en fazla 3, diğer günler 1 takvim günü sonra)."""
    c = get_stock_history(ticker)["Close"]
    r = c.pct_change().loc[start:end]
    gap = pd.Series(c.index, index=c.index).diff().dt.days
    weeks = []
    for _, g in r.groupby(r.index.to_period("W-FRI")):
        gg = gap.loc[g.index]
        if list(g.index.day_of_week) == [0, 1, 2, 3, 4] and gg.iloc[0] <= 3 and (gg.iloc[1:] <= 1).all():
            weeks.append(g.to_numpy())
    W = np.array(weeks)
    cond = (W[:, :3] < 0).all(axis=1)
    out = np.prod(1 + W[:, 3:], axis=1) - 1
    return dict(n_weeks=len(W), n=int(cond.sum()), a=out[cond], b=out[~cond],
                full=int((cond & (W[:, 3:] > 0).all(axis=1)).sum()))


def test_q2_week_pattern_matches_independent_recount():
    with T.data_ceiling(CEIL):
        err, text = call("test_weekday_pattern", Q2)                                # MCP üzerinden oos_start
    assert not err, text
    out = json.loads(text)
    assert out["end"] == CEIL and out["oos_start"] == "2026-01-01" and out["tests_in_this_call"] == 1
    for key, (s, e) in {"in_sample": ("2024-01-01", "2025-12-31"), "out_of_sample": ("2026-01-01", CEIL)}.items():
        x, ref = out[key][LABEL], recount("THYAO", s, e)
        assert x["n_weeks"] == ref["n_weeks"] and x["n"] == ref["n"] and x["full_pattern_count"] == ref["full"]
        assert x["mean"] == pytest.approx(ref["a"].mean() * 100)
        assert x["median"] == pytest.approx(np.median(ref["a"]) * 100)
        assert x["unconditional"] == pytest.approx(ref["b"].mean() * 100)
        assert x["hit_rate"] == pytest.approx((ref["a"] > 0).mean())
        assert x["mean_after_cost"] == pytest.approx((ref["a"].mean() - 2 * COST_ONE_WAY) * 100)   # gidiş-dönüş %0,30
    st = out["stability"][LABEL]                       # 2024-25 anlamlı değil, 2026'da 3 hafta ve ters yön
    assert st["verdict"] == "kararsız" and len(st["reasons"]) == 3
    assert out["out_of_sample"][LABEL]["warning"] == "az gözlem"


def test_week_pattern_ignores_data_after_ceiling(monkeypatch):
    with T.data_ceiling(CEIL):
        a = T.test_weekday_pattern(**Q2)
    full = T._calendar_data()
    monkeypatch.setattr(T, "_calendar_data", lambda: {k: (v.loc[:CEIL] if hasattr(v, "loc") else v)
                                                      for k, v in full.items()})  # 10.09.2026 sonrası veri yok
    b = T.test_weekday_pattern(**Q2)
    assert {k: v for k, v in a.items() if k not in ("data_ceiling", "date_warnings")} == b


@pytest.mark.parametrize("pattern, cond, out", [("week:DDD-UU", [(0, "D"), (1, "D"), (2, "D")], [(3, "U"), (4, "U")]),
                                                ("week:D...-U", [(0, "D")], [(4, "U")]),
                                                ("week:d-u...", [(0, "D")], [(1, "U")]),
                                                ("week:.U-DD.", [(1, "U")], [(2, "D"), (3, "D")])])
def test_week_pattern_grammar(pattern, cond, out):
    assert T._parse_week(pattern) == (cond, out)


@pytest.mark.parametrize("pattern", ["week:DDDUU", "week:...-UU", "week:DD-U", "week:DXD-UU", "week:DDD-.U", "hafta"])
def test_week_pattern_rejects_bad_grammar(pattern):
    with pytest.raises(ToolError, match="pattern"):
        T.test_weekday_pattern(ticker="THYAO", pattern=pattern)


def test_stability_rule():
    ok_is = dict(p=0.01, diff_pt=0.5, n=40)
    assert T._stability(ok_is, dict(n=12, diff_pt=0.2))["verdict"] == "kararlı"
    assert T._stability(ok_is, dict(n=5, diff_pt=0.2))["reasons"] == ["örneklem dışında az gözlem (5 < 10)"]
    assert T._stability(ok_is, dict(n=12, diff_pt=-0.2))["reasons"] == ["örneklem dışında fark ters yönde"]
    assert T._stability(dict(n=4, warning="az gözlem"), dict(n=12, diff_pt=0.2))["verdict"] == "kararsız"


def test_oos_start_checks_and_old_output_kept():
    with pytest.raises(ToolError, match="oos_start"):
        T.test_weekday_pattern(ticker="THYAO", start="2024-01-01", end="2025-12-31", oos_start="2026-03-01")
    with T.data_ceiling(CEIL):
        with pytest.raises(ToolError, match="oos_start"):
            T.test_weekday_pattern(ticker="THYAO", start="2024-01-01", oos_start="2026-10-01")
    w = T.test_weekday_pattern(ticker="THYAO", start="2024-01-01", end="2025-12-31")      # oos_start yok: eski biçim
    assert set(w["results"]) == {"Pzt", "Sal", "Çar", "Per", "Cum"} and w["oos_start"] is None
    assert "in_sample" not in w and "stability" not in w
    s = T.test_weekday_pattern(ticker="THYAO", pattern="down_streak:3", start="2024-01-01", end=CEIL,
                               oos_start="2026-01-01")
    assert list(s["stability"]) == ["down_streak:3 sonrası ertesi gün"]
