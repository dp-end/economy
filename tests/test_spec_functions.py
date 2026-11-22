"""Adım 6: brief §32'deki 28 fonksiyon adı çağrılabilir ve projedeki karşılığıyla aynı sonucu verir."""
import re
from pathlib import Path

import pandas as pd

import src.spec_functions as S

SPEC = Path(__file__).resolve().parent.parent / "README.md"


def test_all_28_names_exist_and_are_mapped():
    assert len(S.SPEC_MAP) == 28
    for name in S.SPEC_MAP:
        assert callable(getattr(S, name)), name


def test_readme_lists_every_name():
    text = SPEC.read_text(encoding="utf-8")
    for name in S.SPEC_MAP:
        assert f"`{name}`" in text, name


def test_wrappers_return_project_results():
    from src.events.technical_events import event_table, states
    from src.indicators.features import full_features
    st = states(full_features("ASELS"))
    assert S.calculate_kama_state("ASELS")["kama_state"].equals(st["kama_state"])
    assert S.calculate_ichimoku_state("ASELS")["cloud_pos"].equals(st["cloud_pos"])
    bb = S.calculate_bollinger_events("ASELS")
    assert set(bb["event"]) <= {"bb_lower_reentry", "bb_upper_reentry", "bb_mid_hold_up", "bb_mid_hold_down"}
    rs = S.calculate_reversal_strength("ASELS", "kama_turn_up")
    e = event_table("ASELS")
    e = e[e["event"] == "kama_turn_up"]
    assert (rs["strength_5d"] - e["direction"] * e["fwd_5"] * 100).abs().max() < 1e-9
    lag = S.rank_sector_laggards("2026-11-06")
    assert lag["lag"].is_monotonic_increasing and lag["ticker"].nunique() == 34
    assert len(S.align_financial_disclosure_dates("2024-01-05")) < len(S.align_financial_disclosure_dates())


def test_news_wrapper_respects_asof():
    ev = S.extract_news_events("ASELS", asof="2026-10-01")
    assert len(ev) and (pd.to_datetime(ev["event_day"]) <= "2026-10-01").all()
