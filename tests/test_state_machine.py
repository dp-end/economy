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
    r = check_numbers("USD/TRY 49,95", {"usdtry": 49.9589})
    assert r["unsupported"] == [] and r["rounding"] == ["49,95"]                 # kesme = yuvarlama hatası, uydurma değil
    r = check_numbers("Net kâr 11,2 milyar TL, özkaynak 335,0 milyar", {"net_income": 11227246450.9, "equity": 334974960000.0})
    assert r["unsupported"] == [] and r["rounding"] == []                       # milyar çevirmesi tanınır


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


def test_thought_signature_is_sent_back():
    """Gemini 3: araç çağrısındaki thought_signature bir sonraki istekte assistant mesajında geri gitmeli."""
    llm = MockLLM(request_tool=("detect_reversal_events", {"ticker": "ASELS", "start": "2026-10-01",
                                                           "end": "2026-11-06"}))
    run_harness("ASELS", "2026-11-06", llm, REVIEW, log=False)
    second = llm.log[1]["messages"]
    asst = [m for m in second if m["role"] == "assistant"][0]
    assert asst["tool_calls"][0]["extra_content"] == {"google": {"thought_signature": "MOCK_SIG"}}


@pytest.mark.parametrize("code,retried", [("503", True), ("500", True), ("429", True), ("401", False),
                                          ("conn", True), ("timeout", True)])
def test_gemini_retries_only_transient_errors(tmp_path, monkeypatch, code, retried):
    """503/500/429 beklenip yeniden denenir (bekleme gecikmeye sayılmaz); 401 gibi kalıcı hata hemen yükselir."""
    from types import SimpleNamespace
    import src.agents.llm as L
    monkeypatch.setattr(L.time, "sleep", lambda s: None)
    calls = []

    def create(**kw):
        calls.append(1)
        if len(calls) == 1:
            import openai
            import httpx2
            req = httpx2.Request("POST", "https://x")
            if code == "conn":
                raise openai.APIConnectionError(request=req)          # "Server disconnected" durumu
            if code == "timeout":
                raise openai.APITimeoutError(request=req)
            raise RuntimeError(f"Error code: {code}")
        msg = SimpleNamespace(content="tamam", tool_calls=None)
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])

    llm = object.__new__(L.GeminiLLM)                       # anahtar okumadan kur
    llm.model, llm.temperature, llm.cache, llm.calls, llm.cache_hits = "test", 0.0, tmp_path, 0, 0
    llm.client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    if retried:
        r = llm.chat([dict(role="user", content="x")])
        assert r["content"] == "tamam" and len(calls) == 2 and r["latency"] < 5
    else:
        with pytest.raises(RuntimeError):
            llm.chat([dict(role="user", content="x")])
        assert len(calls) == 1


def test_memory_never_shows_future_decisions(tmp_path):
    """Adım 14'te bulunan hata: 2023 analizi 2026'da verilmiş insan kararını 'geçmiş karar' diye görüyordu."""
    db = tmp_path / "log.sqlite"
    run_harness("ASELS", "2026-11-06", MockLLM(), REVIEW, db=db)                     # 2026 kararı kaydedildi
    assert len(past_cases("ASELS", db=db)) == 1
    assert len(past_cases("ASELS", db=db, before="2023-02-15")) == 0               # 2023'ten bakınca görünmez
    assert len(past_cases("ASELS", db=db, before="2026-11-07")) == 1               # sonrasından bakınca görünür
    llm = MockLLM()
    run_harness("ASELS", "2023-02-15", llm, REVIEW, db=db)
    sent = " ".join(m.get("content") or "" for call in llm.log for m in call["messages"])
    assert "2026-11-06" not in sent                                                 # modele gelecek tarih gitmedi


def test_validator_skips_window_lengths_but_not_values():
    """'20, 50 ve 200 günlük' pencere uzunluğudur, uydurma sayı değil (Adım 15'te B sürümünde bulunan yanlış alarm)."""
    from src.agents.validate import check_numbers, numbers_in
    raw = lambda t: [r for _, _, r in numbers_in(t)]  # noqa: E731
    assert raw("fiyat 20, 50 ve 200 günlük ortalamaların altında, RSI 37,99") == ["37,99"]
    assert raw("the 50-day average is 12.5; 10 işlem gününde %-15.9") == ["12.5", "-15.9"]
    assert raw("%20 düştü, 20 hisse") == ["20", "20"]                     # gerçek değerler korunur
    r = check_numbers("20 günlük ortalama 342,5 TL", {"sma_20": 342.525})
    assert r["n_numbers"] == 1 and r["unsupported"] == []


def test_validator_skips_indicator_parameters_but_not_values():
    """'RSI(14)', 'SMA 20, 50, 200' gösterge parametresidir (son deneyde B sürümünde bulunan yanlış alarm);
    'RSI 33,88', 'EMA 4,10 TL' gibi değerler yine kontrol edilir."""
    from src.agents.validate import numbers_in
    raw = lambda t: [r for _, _, r in numbers_in(t)]  # noqa: E731
    assert raw("RSI(14) 26,53 seviyesinde") == ["26,53"]
    assert raw("fiyat tüm hareketli ortalamaların (SMA 20, 50, 200) altında") == []
    assert raw("SMA200 üstünde, EMA 20 ve 50 kesişti, MACD(12, 26, 9)") == []
    assert raw("RSI 33,88 ile, 20 günlük EMA 4,10 TL") == ["33,88", "4,10"]       # değerler korunur
    assert raw("RSI 26, MACD negatif") == ["26"]                                   # parantezsiz RSI = değer
    assert raw("RSI (37,99) ve MACD(12,26,9)") == ["37,99"]                         # Türkçe ondalık = değer
