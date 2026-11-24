"""Yerel model 2. adım: soru modu (src/harness/question_mode.py). Model yerine ScriptLLM; araçlar gerçek (MCP)."""
import json

import pytest

from src.agents.llm import ScriptLLM
from src.harness import question_mode as QM
from src.mcp_server import tools as T

ASOF = "2026-09-10"
Q1 = ("Perakende sektöründe (BIMAS, MGROS, MAVI) 10.09.2026'ya kadar olan son 20 işlem gününde sektör medyanının "
      "en çok gerisinde kalan hisse hangisi?")
RANK = ("rank_sector_relative_strength", dict(tickers=["BIMAS", "MGROS", "MAVI"], horizon=20, date=ASOF))
GOOD = ("- MAVI son 20 günde kendisi hariç akran ortancasının %11,26 gerisinde; en geride kalan hisse.\n"
        "- BIMAS aynı dönemde %14,21 yükseldi.\n- Tek dönemlik fark; geçmiş kapanma oranı ayrıca test edilmeli.\n"
        "ANALİTİK DURUM: WATCH")


def run(steps, question=Q1, **kw):
    llm = ScriptLLM(steps)
    return QM.run_question(question, ASOF, llm, log=False, **kw), llm


def test_happy_path_tool_then_supported_answer():
    res, llm = run([{"tool_calls": [RANK]}, {"content": GOOD}])
    assert res["action"] == "WATCH" and res["action_source"] == "model" and res["status"] == "ok"
    assert res["retries"] == 0 and res["invalid_tool_calls"] == 0 and res["validation"]["numbers"]["unsupported"] == []
    assert len(llm.log[0]["tools"]) == 15                                       # CALL_TOOLS: 15 araç açık
    tool_msg = json.loads(llm.log[1]["messages"][-1]["content"])                # modelin gördüğü (kısaltılmış) çıktı
    assert tool_msg["most_behind"] == "MAVI" and tool_msg["data_ceiling"] == ASOF
    assert "Analiz tarihi: 2026-09-10" in llm.log[0]["messages"][0]["content"]


def test_made_up_number_fixed_on_retry():
    res, llm = run([{"tool_calls": [RANK]}, {"content": GOOD.replace("WATCH", "WATCH") + "\nBeklenen getiri 987,65."},
                    {"content": GOOD}])
    assert res["retries"] == 1 and res["status"] == "ok" and res["action"] == "WATCH"
    assert "987,65" in llm.log[2]["messages"][-1]["content"] and llm.log[2]["tools"] == []   # düzeltme: araçsız


def test_made_up_number_twice_withheld_as_unsafe():
    bad = GOOD + "\nBeklenen getiri 987,65."
    res, _ = run([{"tool_calls": [RANK]}, {"content": bad}, {"content": bad}])
    assert res["status"] == "withheld" and res["action"] == "ANALYSIS_UNSAFE"
    assert "987,65" not in res["answer"].split("(")[0] and "denetimden geçmedi" in res["answer"]
    assert "987,65" in res["answer_raw"]                                         # ham cevap günlükte saklanır


def test_advice_and_target_price_blocked():
    advice = "MAVI %11,26 geride. Hedef fiyat 250 TL, almanızı öneririm.\nANALİTİK DURUM: WATCH"
    res, _ = run([{"tool_calls": [RANK]}, {"content": advice}, {"content": advice}])
    assert res["action"] == "ANALYSIS_UNSAFE" and res["status"] == "withheld"
    assert any("tavsiye" in p for p in res["validation"]["problems"])


@pytest.mark.parametrize("text, hit", [("Yatırım tavsiyesi ve hedef fiyat veremem.", False),
                                       ("Almanızı ya da satmanızı önermiyorum.", False),
                                       ("Hedef fiyat: 52,40 TL", True), ("hedef fiyatı yaklaşık 60", True),
                                       ("Bu hisseyi almanızı öneririm.", True), ("Alım fırsatı var.", True)])
def test_advice_detector(text, hit):
    assert bool(QM.advice_hits(text)) == hit


@pytest.mark.parametrize("text, action", [("...\nANALİTİK DURUM: WATCH", "WATCH"),
                                          ("Analitik durum: investigate", "INVESTIGATE"),
                                          ("ANALITIK DURUM: **REJECT_SIGNAL**", "REJECT_SIGNAL"),
                                          ("ANALİTİK DURUM: BUY", "BUY"), ("eylem yok", None)])
def test_action_line_parsing(text, action):
    assert QM.parse_action(text) == action


def test_action_outside_allowed_set_becomes_unsafe():
    buy = "MAVI %11,26 geride.\nANALİTİK DURUM: BUY"
    res, _ = run([{"tool_calls": [RANK]}, {"content": buy}, {"content": buy}])
    assert res["action"] == "ANALYSIS_UNSAFE" and not res["validation"]["action_valid"]


def test_answer_without_any_tool_is_unsafe():
    res, _ = run([{"content": "Genel olarak perakende güçlü.\nANALİTİK DURUM: WATCH"}])
    assert res["action"] == "ANALYSIS_UNSAFE" and res["action_source"].startswith("kural: başarılı araç")


def test_invalid_calls_reported_back_and_counted():
    res, llm = run([{"tool_calls": [("hisse_al", {}), ("rank_sector_relative_strength", {"sector": "uzay"})]},
                    {"tool_calls": [RANK]}, {"content": GOOD}])
    assert res["invalid_tool_calls"] == 2 and res["action"] == "WATCH"
    msgs = [m["content"] for m in llm.log[1]["messages"] if m["role"] == "tool"]
    assert msgs[0].startswith("HATA: bilinmeyen araç") and "bilinmeyen sektör" in msgs[1]


def test_call_budget_then_forced_answer():
    res, llm = run([{"tool_calls": [RANK] * (QM.MAX_CALLS + 1)}, {"content": GOOD}])
    done = [o for o in res["tool_calls"] if o["ok"]]
    assert len(done) == QM.MAX_CALLS and res["tool_calls"][-1].get("over_budget")
    assert llm.log[1]["messages"][-1]["content"] == QM.FORCE_TEXT and llm.log[1]["tools"] == []   # ANSWER: araç yok
    assert res["action"] == "WATCH"


def test_ceiling_is_set_by_harness_not_model():
    late = ("rank_sector_relative_strength", dict(tickers=["BIMAS", "MGROS", "MAVI"], horizon=20, date="2026-11-06"))
    res, llm = run([{"tool_calls": [late]}, {"content": GOOD}])
    out = res["tool_calls"][0]["output"]
    assert out["date"] == ASOF and out["date_warnings"]                         # model ileri tarih istedi: kırpıldı
    assert T._active_ceiling() is None                                           # tavan soru bitince kalktı


def test_hard_block_overrides_model():
    assert QM.hard_block([dict(ok=True, output={"gate": {"status": "HARD_BLOCK"}})])
    assert not QM.hard_block([dict(ok=True, output={"gate": {"status": "PASS"}}), dict(ok=False, output=None)])


def test_log_and_replay(tmp_path):
    db = tmp_path / "q.sqlite"
    res = QM.run_question(Q1, ASOF, ScriptLLM([{"tool_calls": [RANK]}, {"content": GOOD}]), db=db)
    row = QM.load_question(res["qid"], db)
    assert row["action"] == "WATCH" and row["asof"] == ASOF and row["tool_calls"][0]["name"] == RANK[0]
    assert row["outputs"][0]["most_behind"] == "MAVI"
    rep = QM.replay_question(res["qid"], db)
    assert rep["same_outputs"] and rep["n_calls"] == 1


def test_compact_view_only_selects_and_rounds():
    with T.data_ceiling(ASOF):
        full = T.to_json(T.calculate_fundamental_change(ticker="TUPRS", start="2024-01-01"))
        ev = T.to_json(T.detect_reversal_events(ticker="ASELS", event="bb_lower_touch", start="2024-01-01"))
    c = QM.compact("calculate_fundamental_change", full)
    assert len(c["quarters"]) == len(full["quarters"]) == 11
    for a, b in zip(c["quarters"], full["quarters"]):
        assert a["ebitda_margin_q"] == round(b["ebitda_margin_q"], 4) and a["event_day"] == b["event_day"][:10]
        assert a["reaction"]["ab_xu100_20_pct"] == round(b["reaction"]["ab_xu100_20_pct"], 4)
    e = QM.compact("detect_reversal_events", ev)
    assert e["by_condition"] == QM._round(ev["by_condition"]) and len(e["events"]) == 5 and "events_note" in e
    assert len(json.dumps(c, ensure_ascii=False)) < 8000 and len(json.dumps(e, ensure_ascii=False)) < 8000


def test_model_error_does_not_crash_and_is_logged(tmp_path):
    """Gerçek deneme bulgusu (24 Kasım): sunucu bellek yetmeyince 404 verdi ve soru modu çöktü."""
    llm = ScriptLLM([{"tool_calls": [RANK]}, {"raise": "Error code: 404"}])
    res = QM.run_question(Q1, ASOF, llm, db=tmp_path / "q.sqlite")
    assert res["status"] == "llm_error" and res["action"] == "ANALYSIS_UNSAFE" and "404" in res["answer"]
    assert res["tool_calls"][0]["ok"] and QM.load_question(res["qid"], tmp_path / "q.sqlite")["status"] == "llm_error"
