import pytest

from src.agents.llm import MockLLM
from src.agents.validate import check_numbers
from src.harness.decision_log import load_analysis, load_trace, past_cases
from src.harness.state_machine import MANDATORY, STATE_TOOLS, STATES, allowed_next, run_harness

REVIEW = dict(decision="ACCEPT", reason="test incelemesi")


def test_transitions_fixed_and_safe_path():
    s, seen = "START", []
    while s:
        seen.append(s)
        s = allowed_next(s, hard_blocked=False)
    assert seen == STATES                                                       # sıra sabit
    s, seen = "SELECT_UNIVERSE", []
    while s:
        seen.append(s)
        s = allowed_next(s, hard_blocked=True)
    assert seen == ["SELECT_UNIVERSE", "CHECK_DATA_QUALITY", "RISK_GATE", "GENERATE_EXPLANATION",
                    "HUMAN_REVIEW", "LOG_AND_CLOSE"]                            # engelde bile zorunlular atlanmaz
    assert MANDATORY <= set(seen)
    assert set(STATE_TOOLS) == set(STATES)


def test_full_run_logs_decision_review_and_trace(tmp_path):
    db = tmp_path / "log.sqlite"
    ctx, tr = run_harness("ASELS", "2026-11-06", MockLLM(), REVIEW, db=db)
    assert tr["states"] == STATES
    s = ctx["summary"]
    assert s["invalid_tool_calls"] == 0 and s["tool_errors"] == 0 and s["llm_calls"] == 6
    assert all(c["tool"] in STATE_TOOLS[c["state"]] for c in tr["tool_calls"])      # her çağrı kendi durumunda izinli
    a = load_analysis(ctx["analysis_id"], db)
    assert a["action"] == ctx["record"]["action"] and a["explanation"] == ctx["explanation"]
    assert past_cases("ASELS", db=db).iloc[0]["decision"] == "ACCEPT"
    assert load_trace(ctx["analysis_id"], db)["summary"]["states"] == len(STATES)


def test_forbidden_tool_is_blocked_and_counted():
    llm = MockLLM(request_tool=("run_backtest", {"strategy": "A_teknik"}))      # piyasa ajanı backtest isteyemez
    ctx, tr = run_harness("ASELS", "2026-11-06", llm, REVIEW, log=False)
    assert ctx["summary"]["invalid_tool_calls"] == 1
    bad = tr["invalid_tool_calls"][0]
    assert bad["state"] == "RUN_PATTERN_AND_REVERSAL_ANALYSIS" and bad["by"] == "market"
    assert not any(c["tool"] == "run_backtest" and c["state"] == "RUN_PATTERN_AND_REVERSAL_ANALYSIS"
                   for c in tr["tool_calls"])                                     # çağrı gerçekleşmedi


def test_allowed_tool_request_is_executed():
    llm = MockLLM(request_tool=("detect_reversal_events", {"ticker": "ASELS", "start": "2026-10-01",
                                                           "end": "2026-11-06"}))
    ctx, tr = run_harness("ASELS", "2026-11-06", llm, REVIEW, log=False)
    assert ctx["summary"]["invalid_tool_calls"] == 0
    assert any(c["by"] == "market" and c["tool"] == "detect_reversal_events" for c in tr["tool_calls"])


@pytest.mark.parametrize("fix", [True, False])
def test_hallucinated_number_is_caught(fix):
    ctx, tr = run_harness("ASELS", "2026-11-06", MockLLM(hallucinate=True, fix_on_retry=fix), REVIEW, log=False)
    o = [x for x in tr["llm"] if x["agent"] == "orchestrator"][0]
    assert o["retried"] and "987,65" in o["unsupported"]
    assert ctx["explanation_verified"] is fix
    assert ctx["explanation"].startswith("[DOĞRULANMAMIŞ") is (not fix)
    assert ctx["summary"]["unsupported_numbers"] > 0                             # ajan notlarındaki uydurma sayılır


def test_unknown_ticker_safe_failure():
    ctx, tr = run_harness("YOKBOYLE", "2026-11-06", MockLLM(), REVIEW, log=False)
    assert "CHECK_DATA_QUALITY" in tr["states"] and "BUILD_EVIDENCE" not in tr["states"]
    assert ctx["record"]["action"] == "ANALYSIS_UNSAFE" and ctx["gate"]["status"] == "HARD_BLOCK"


def test_validator_handles_turkish_numbers():
    ctx = ["E1 net kâr değişimi %81.6, faiz %36.0", {"toplam_getiri": 2.4137}]
    assert check_numbers("Net kâr %81,6 arttı (E1); getiri %241.", *ctx)["unsupported"] == []
    assert check_numbers("Hedef fiyat 1.234,5 TL", *ctx)["unsupported"] == ["1.234,5"]


def test_validator_skips_written_dates_and_years():
    text = "Şirket 7, 8, 23, 26 Ekim ve 3 Kasım 2026 tarihlerinde sözleşme duyurdu; 28 Ekim 2026'da Fed %3,75-%4 dedi."
    r = check_numbers(text, {"konusmadaki_deger": [3.75, 4.0]})
    assert r["unsupported"] == [] and r["n_numbers"] == 1                       # yalnızca 3,75 sayıldı
    assert check_numbers("USD/TRY 49,95", {"usdtry": 49.9589})["unsupported"] == ["49,95"]   # yanlış yuvarlama yakalanır


def test_backtest_tool_computes_benchmark_differences():
    from src.mcp_server.tools import run_backtest
    p = run_backtest("C_temel", end="2026-11-06")["performance"]
    assert abs(p["xu100_farki"] - (p["toplam_getiri"] - p["xu100_getiri"])) < 1e-12
    assert abs(p["esit_agirlik_farki"] - (p["toplam_getiri"] - p["esit_agirlik_getiri"])) < 1e-12
    assert p["xu100_farki"] > 0 > p["esit_agirlik_farki"]                      # C: XU100'ün üstünde, al-tut'un altında


def test_history_described_with_explicit_names():
    from src.harness.state_machine import describe_history
    h = describe_history({"fundamental": dict(n=162, horizon=5, last_date="2026-10-30", mean_top=0.0074,
                                              mean_bottom=-0.0123, spread=0.0197),
                          "tech:kama_turn_up": dict(n=10, horizon=5, last_date="x", hit=0.47, mean_exc=0.001)})
    assert set(h) == {"bilanco_ucte_bir_tepkisi", "teknik_olay_kama_turn_up"}
    assert h["bilanco_ucte_bir_tepkisi"]["ust_ucte_bir_ortalama_fazla_getiri"] == 0.0074
