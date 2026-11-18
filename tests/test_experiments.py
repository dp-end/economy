import pandas as pd

from src.agents.llm import MockLLM
from src.harness.experiments import (evidence_completeness, future_call, reference, run_A, unsupported_conclusion)
from src.harness.state_machine import VERSION_STATES, allowed_next, parse_answer, reason_hint
from src.harness.evidence import run_analysis


def test_parse_answer():
    assert parse_answer('Yanıt: {"action": "watch", "explanation": "x"}') == ("WATCH", "x")
    assert parse_answer('{"action": "BUY", "explanation": "al"}')[0] == "GECERSIZ"     # sınırlı eylem kümesi
    assert parse_answer("düz metin")[0] == "GECERSIZ"


def test_unsupported_conclusion_rules():
    ref = dict(gate="HARD_BLOCK", conditions={})
    assert unsupported_conclusion("WATCH", ref) and not unsupported_conclusion("ANALYSIS_UNSAFE", ref)
    ref = dict(gate="PASS", conditions=dict(shock=True, catch_up=True))
    assert unsupported_conclusion("POTENTIAL_CATCH_UP_CANDIDATE", ref)                # şokta aday
    ref = dict(gate="PASS", conditions=dict(fundamental=False))
    assert unsupported_conclusion("FUNDAMENTAL_IMPROVEMENT_CANDIDATE", ref)            # koşulu yok
    assert not unsupported_conclusion("WATCH", ref) and unsupported_conclusion("GECERSIZ", ref)


def test_future_call_detection():
    assert future_call("get_stock_history", {"ticker": "ASELS"}, "2023-02-15")         # tarih yok = son gün
    assert not future_call("get_stock_history", {"end": "2023-02-15"}, "2023-02-15")
    assert future_call("get_fundamental_metrics", {"asof": "2024-01-01"}, "2023-02-15")
    assert not future_call("get_speech_features", {}, "2023-02-15")


def test_evidence_completeness_and_reference():
    ref = reference("ASELS", "2026-11-06")
    assert ref["action"] == "FUNDAMENTAL_IMPROVEMENT_CANDIDATE" and ref["committed"]
    full = " ".join(f"({e['id']})" for e in ref["committed"])
    assert evidence_completeness(full, ref) == 1.0
    assert evidence_completeness("Net kâr %81,6 arttı.", ref) > 0                       # sayıyla karşılanan kanıt
    assert evidence_completeness("hiçbir şey", ref) == 0.0
    assert evidence_completeness("x", reference("YOKBOYLE", "2026-11-06")) is None


def test_raw_version_uses_point_in_time_text():
    r = run_A(MockLLM(), "ASELS", "2023-02-15")
    assert r["action"] == "WATCH" and r["tool_calls"] == 0 and r["llm_calls"] == 1


def test_version_state_lists_and_hint():
    assert "BUILD_EVIDENCE" not in VERSION_STATES["C"] and "RISK_GATE" not in VERSION_STATES["C"]
    assert "HUMAN_REVIEW" not in VERSION_STATES["D"] and VERSION_STATES["E"][-1] == "LOG_AND_CLOSE"
    assert allowed_next("GENERATE_EXPLANATION", False, "C") is None
    assert allowed_next("SELECT_UNIVERSE", True, "C") == "LOAD_MARKET_DATA"             # C'de engel yok
    hint = reason_hint(run_analysis("ASELS", "2026-11-06"))
    assert "E1" in hint and "kanıt" in hint
