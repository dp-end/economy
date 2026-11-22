import copy
import itertools

import pandas as pd
import pytest

import src.harness.evidence as evidence
from src.harness.decision_log import (load_analysis, log_decision, past_cases, record_review, replay_analysis)
from src.harness.evidence import ACTIONS, analysis_day, build_evidence_bundle, decide, run_analysis
from src.harness.quality_gate import GATE_OUTPUTS, check_data_quality

MCP_TOOLS = {"get_stock_history", "get_index_history", "calculate_indicators", "detect_reversal_events",
             "find_support_resistance", "rank_sector_relative_strength", "test_weekday_pattern",
             "get_fundamental_metrics", "calculate_fundamental_change", "get_macro_context", "get_news_events",
             "get_speech_features", "run_backtest", "check_data_quality", "build_evidence_bundle"}


@pytest.fixture(scope="module")
def asels():
    return run_analysis("ASELS", "2026-11-06")


def test_action_never_leaves_finite_set():
    keys = ["laggard", "sector_strong", "recovering", "weak_earnings", "catch_up", "fundamental", "technical",
            "shock", "negative_news", "tech_conflict"]
    for vals in itertools.product([False, True], repeat=len(keys)):
        for gate in GATE_OUTPUTS:
            action, secondary = decide(dict(zip(keys, vals)), gate)
            assert action in ACTIONS and set(secondary) <= set(ACTIONS)


def test_decide_rules():
    base = dict(laggard=False, catch_up=False, fundamental=False, technical=False, shock=False, negative_news=False)
    assert decide({**base, "fundamental": True}, "HARD_BLOCK")[0] == "ANALYSIS_UNSAFE"
    assert decide({**base, "catch_up": True, "shock": True}, "PASS")[0] == "REJECT_SIGNAL"
    assert decide({**base, "negative_news": True}, "PASS")[0] == "INVESTIGATE"
    assert decide({**base, "catch_up": True, "technical": True}, "PASS") == (
        "POTENTIAL_CATCH_UP_CANDIDATE", ["TECHNICAL_REVERSAL_CANDIDATE"])
    assert decide({**base, "laggard": True}, "PASS")[0] == "WATCH"
    assert decide(base, "PASS")[0] == "HOLD"


def test_corrupted_bundle_hard_block(asels):
    b = copy.deepcopy(asels)
    asof = pd.Timestamp(b["meta"]["asof"])
    b["source_dates"]["bilanco_event_day"] = asof + pd.Timedelta(days=30)        # gelecekteki bilanço
    g = check_data_quality(b)
    assert g["status"] == "HARD_BLOCK" and any("gelecek bilgisi" in h for h in g["hard_blocks"])
    b = copy.deepcopy(asels)
    b["observed"]["fundamentals"]["period_end"] = asof + pd.Timedelta(days=60)
    assert check_data_quality(b)["status"] == "HARD_BLOCK"                        # dönem uyumsuzluğu
    b = copy.deepcopy(asels)
    b["observed"]["price"]["max_abs_ret_250"] = 0.5                               # bozuk fiyat (bölünme vb.)
    assert check_data_quality(b)["status"] == "HARD_BLOCK"
    b = copy.deepcopy(asels)
    b["backtest"]["end"] = b["source_dates"]["backtest_bitis"] = asof + pd.Timedelta(days=5)
    assert check_data_quality(b)["status"] == "HARD_BLOCK"                        # backtest gelecek veriyi kullanmış


def test_future_statement_never_enters_bundle(monkeypatch):
    """Tabloya sonradan yayınlanacak sahte bir bilanço eklense bile o günün paketine giremez."""
    t = dict(evidence._tables())
    fake = t["earnings"][t["earnings"]["ticker"] == "ASELS"].tail(1).copy()
    fake["period_end"], fake["event_day"], fake["tercile"] = pd.Timestamp("2026-12-31"), pd.Timestamp("2027-02-20"), "alt"
    t["earnings"] = pd.concat([t["earnings"], fake], ignore_index=True)
    monkeypatch.setattr(evidence, "_tables", lambda: t)
    b = build_evidence_bundle("ASELS", "2026-11-06")
    assert pd.Timestamp(b["observed"]["fundamentals"]["period_end"]) == pd.Timestamp("2026-09-30")
    assert check_data_quality(b)["status"] != "HARD_BLOCK"


def test_point_in_time_fundamentals():
    before = build_evidence_bundle("ASELS", "2026-10-27")["observed"]["fundamentals"]
    after = build_evidence_bundle("ASELS", "2026-10-28")["observed"]["fundamentals"]
    assert pd.Timestamp(before["period_end"]) == pd.Timestamp("2026-06-30")       # 2026/9 bilançosu 27 Ekim 18:34'te
    assert pd.Timestamp(after["period_end"]) == pd.Timestamp("2026-09-30")        # ertesi işlem günü görünür


def test_unknown_ticker_and_quake_are_safe_failures():
    r = run_analysis("YOKBOYLE", "2026-11-06")
    assert r["gate"]["status"] == "HARD_BLOCK" and r["action"] == "ANALYSIS_UNSAFE" and r["committed"] == []
    q = run_analysis("ASELS", "2023-02-15")
    assert q["gate"]["status"] == "ANALYSIS_UNSAFE" and q["action"] == "ANALYSIS_UNSAFE"


def test_bundle_dates_and_evidence(asels):
    asof = pd.Timestamp(asels["meta"]["asof"])
    assert pd.Timestamp(asels["backtest"]["end"]) <= asof
    assert all(pd.Timestamp(h["last_date"]) <= asof for h in asels["history"].values() if h)
    ids = [e["id"] for e in asels["committed"]]
    assert ids == [f"E{i}" for i in range(1, len(ids) + 1)]
    assert all(e["text"] and e["tool"] in MCP_TOOLS for e in asels["committed"])  # her kanıt bir araca dayanır
    assert analysis_day("2026-11-08") == pd.Timestamp("2026-11-06")               # Pazar -> Cuma


def test_shock_evidence_has_direction_and_month_return():
    """Şok kanıtı yönü (düşüş/yükseliş), günü ve 20 günlük hisse/XU100 getirisini söylemeli (Adım 14 bulgusu)."""
    k = run_analysis("KRDMD", "2026-11-06")
    p = k["observed"]["price"]
    assert k["conditions"]["shock"] and p["shock_ret"] <= -0.095
    assert pd.Timestamp(p["shock_day"]) == pd.Timestamp("2026-11-05")
    assert p["ret_20d"] < p["xu100_ret_20d"] < 0                                   # hisse endeksten çok düştü
    shock = [e for e in k["committed"] if e["text"].startswith("Şok bayrağı")]
    assert len(shock) == 1 and "2026-11-05" in shock[0]["text"] and "düşüş" in shock[0]["text"]
    assert "XU100" in shock[0]["text"] and shock[0]["value"] == p["shock_ret"]


def test_decision_log_review_and_replay(tmp_path, asels):
    db = tmp_path / "log.sqlite"
    aid = log_decision(asels, db=db)
    record_review(aid, "MODIFY", "Backtest al-tut'u geçmiyor.", final_action="WATCH", db=db)
    with pytest.raises(ValueError):
        record_review(aid, "BUY", "x", db=db)                                     # sınırlı karar kümesi
    with pytest.raises(ValueError):
        record_review(aid, "MODIFY", "x", final_action="BUY_NOW", db=db)         # sınırlı eylem kümesi
    with pytest.raises(ValueError):
        record_review(aid, "ACCEPT", "  ", db=db)                                 # gerekçe zorunlu
    stored = load_analysis(aid, db)
    assert stored["action"] == asels["action"] and stored["committed"][0]["id"] == "E1"
    r = replay_analysis(aid, db)
    assert r["same"] and r["new_action"] == r["stored_action"] and not r["procedure_changed"]
    cases = past_cases("ASELS", db=db)
    assert len(cases) == 1 and cases.iloc[0]["final_action"] == "WATCH"


def test_profit_direction_words():
    """Yüzde tek başına yanıltır (PETKM: zarar küçüldü ama +%48); cümle yönü sözle söylemeli."""
    from src.harness.evidence import profit_direction
    assert profit_direction(11.2e9, 6.2e9) == "kâr arttı"
    assert profit_direction(1e9, 2e9) == "kâr azaldı"
    assert profit_direction(-5.4e9, 5.0e9) == "kârdan zarara geçti"           # TURSG
    assert profit_direction(0.27e9, -0.88e9) == "zarardan kâra geçti"         # KRDMD
    assert profit_direction(-0.82e9, -1.58e9) == "zarar küçüldü (hâlâ zarar)"  # PETKM
    assert profit_direction(-2e9, -1e9) == "zarar büyüdü"
    assert profit_direction(None, 1e9) is None
    p = run_analysis("PETKM", "2026-11-06")
    text = next(e["text"] for e in p["committed"] if e["text"].startswith("Son açıklanan"))
    assert "zarar küçüldü" in text and "bir yıl önce" in text


def test_tech_history_is_direction_adjusted():
    """Ayı olayında başarı = hisse XU100'ün gerisinde kalır. Yön yok sayılırsa oran ters çıkar (Adım 15 bulgusu)."""
    from src.harness.evidence import shift_days, tech_history
    day = pd.Timestamp("2026-11-06")
    te = pd.read_parquet("reports/technical_events.parquet")
    g = te[(te["event"] == "kama_turn_down") & (te["date"] <= shift_days(day, 5)) & ~te["quake_window"]]["exc_5"].dropna()
    h = tech_history("kama_turn_down", day)
    assert h["direction"] == -1 and h["n"] == len(g)
    assert abs(h["hit"] - (g < 0).mean()) < 1e-12 and abs(h["mean_exc"] + g.mean()) < 1e-12
    up = tech_history("kama_turn_up", day)
    assert up["direction"] == 1 and up["hit"] < 0.5                     # boğa olayı değişmez: %46,7


def test_conflicting_technical_signals_mean_investigate():
    """Son 3 günde hem boğa hem ayı teknik olay -> INVESTIGATE; şok/haberli aday yine REJECT_SIGNAL (Adım 15 kararı)."""
    base = dict(laggard=False, catch_up=False, fundamental=False, technical=False, shock=False, negative_news=False)
    assert decide({**base, "tech_conflict": True}, "PASS") == ("INVESTIGATE", [])
    assert decide({**base, "fundamental": True, "tech_conflict": True}, "PASS") == \
        ("INVESTIGATE", ["FUNDAMENTAL_IMPROVEMENT_CANDIDATE"])
    assert decide({**base, "fundamental": True, "shock": True, "tech_conflict": True}, "PASS")[0] == "REJECT_SIGNAL"
    p = run_analysis("PETKM", "2026-11-06")                       # kama_turn_up + kama_turn_down + support_break
    assert p["conditions"]["tech_conflict"] and p["action"] == "INVESTIGATE"
    txt = next(e["text"] for e in p["committed"] if e["text"].startswith("Çelişen teknik"))
    assert "boğa kama_turn_up" in txt and "support_break" in txt


def test_speech_after_close_is_not_known_the_same_day():
    """Fed basın toplantısı 2026-10-28 21:30'da (BIST kapandıktan sonra): 28 Ekim analizi görmemeli, 30 Ekim görmeli."""
    from src.harness.evidence import observe
    same = observe("ASELS", pd.Timestamp("2026-10-28"))[0]["speech"]["sources"]
    later = observe("ASELS", pd.Timestamp("2026-10-30"))[0]["speech"]["sources"]
    assert "fed_20261028" not in [s["source_id"] for s in same]
    assert "fed_20261028" in [s["source_id"] for s in later]


def test_fundamental_evidence_in_report_money_and_unit():
    """Geçmiş tarihli analizde kâr, o raporun parasıyla ve birimi yazılarak verilir (Adım 1b)."""
    from src.harness.evidence import unit_tr
    assert unit_tr("TL@2026-09") == "Eylül 2026 TL'si" and unit_tr("TL nominal") == "nominal TL"
    assert unit_tr(None) == "" and unit_tr("-") == ""
    fu = build_evidence_bundle("ASELS", "2024-06-03")["observed"]["fundamentals"]
    assert fu["period_end"] == pd.Timestamp("2024-03-31") and fu["unit"] == "TL@2024-03"
    assert abs(fu["net_income"] - 1.403767e9) / 1.403767e9 < 1e-4          # Mart 2024 TL'si, açıklandığı gibi
    assert fu["tercile"] is None                     # 2024/1: önceki çeyrekte (2023/4) yeterli karşılaştırma yok
    r = run_analysis("ASELS", "2026-11-06")
    e1 = next(e["text"] for e in r["committed"] if e["text"].startswith("Son açıklanan bilanço"))
    assert "ikisi de Eylül 2026 TL'si" in e1 and "bir önceki çeyrekte 30 hissenin" in e1


# ---------- Adım 6: BIST 100 üyelik kontrolü (brief §5) ----------
def test_index_membership_lookup():
    from src.data.universe import index_membership
    m = index_membership("ASELS", "2026-11-06")
    assert m["member"] is True and m["period"] == "2026-10-01/2026-12-31" and "kap.org.tr" in m["source"]
    assert index_membership("ASELS", "2024-01-05")["member"] is None          # dönem doğrulanmadı
    assert index_membership("SKBNK", "2026-11-06")["member"] is None          # dosya tam liste değil


def test_gate_warns_only_when_membership_is_false():
    from src.harness.quality_gate import check_data_quality
    from src.harness.evidence import build_evidence_bundle
    b = build_evidence_bundle("ASELS", "2026-11-06")
    base = check_data_quality(b)["soft_warnings"]
    b["meta"]["bist100"] = dict(member=None)
    assert check_data_quality(b)["soft_warnings"] == base                     # bilinmeyen dönem uyarı değil
    b["meta"]["bist100"] = dict(member=False)
    assert any("BIST 100 üyesi değil" in w for w in check_data_quality(b)["soft_warnings"])
