"""Yerel model 5. adım: görülmemiş soru seti (src/finetune/heldout.py) ve puanlama (src/finetune/evaluate.py)."""
import asyncio
import json

import pytest

from src.agents.llm import ScriptLLM
from src.finetune import build_dataset as B
from src.finetune import evaluate as E
from src.finetune import heldout as H
from src.finetune import templates as TP
from src.mcp_server import tools as TOOLS

SLOTS = dict(grup="Perakende", tickers="BIMAS, MGROS", asof="10.09.2026", h=20, start="01.01.2024", bas="2024 başından",
             T="THYAO", is_start="01.01.2024", is_end="31.12.2025", oos_start="01.01.2026", desc="düzen", k=3,
             yon_abl="düşüşten", yon_gen="düşüşün", yon_ad="düşüş", yon_sonra="düştükten", event_tr="olay",
             yon="yükseldi", cond_q="", strat_tr="A stratejisi", bad="XYZAB", future="01.12.2026")
SMALL_PLAN = [("s1", dict(variant="rank")), ("s2", dict(kind="weekday")), ("s4", dict(bank=False)),
              ("s5", dict(kind="plain")), ("other", None), ("error", None)]


# ---------------------------------------------------------------- soru seti
def test_heldout_phrasings_are_new_and_complete():
    assert set(H.Q_EVAL) == set(TP.Q)
    train = {t for ts in TP.Q.values() for t in ts}
    for key, ts in H.Q_EVAL.items():
        for t in ts:
            assert t not in train, t
            t.format(**SLOTS)                                     # yuva adları örnekleyicinin verdikleri
    assert len(H.Q_EVAL["o_ind"]) == 2                           # çift/tek sıra = hangi göstergeler (Sampler.other)
    for new, old in ((H.WEEK_EVAL, TP.WEEK_PATTERNS), (H.EVENT_EVAL, TP.EVENT_TR), (H.STRAT_EVAL, TP.STRAT_TR)):
        assert list(new) == list(old) and all(new[k] != old[k] for k in old)
    for d in (1, -1):                                            # koşul kodları aynı, ifadeleri farklı
        assert [c for _, c in H.COND_EVAL[d]] == [c for _, c in B.COND_SETS[d]]
        assert all(a != b for (a, _), (b, _) in zip(H.COND_EVAL[d][1:], B.COND_SETS[d][1:]))
    assert H.yazili("2026-09-10") == "10 Eylül 2026"


def _as_training(rows):
    """Soru seti satırlarını eğitim konuşması biçimine çevirir (çakışma denetimini sınamak için)."""
    for r in rows:
        msgs = [dict(role="assistant", content=None, tool_calls=[
            dict(id=f"c{i}", type="function", function=dict(name=c["name"], arguments=json.dumps(c["args"])))
            for i, c in enumerate(r["gold_calls"])])]
        msgs += [dict(role="tool", tool_call_id=f"c{i}", content="{}" if c["ok"] else "HATA: x")
                 for i, c in enumerate(r["gold_calls"])]
        yield dict(cat=r["cat"], asof=r["asof"], question=r["question"], messages=msgs)


def test_heldout_build_is_reproducible_and_avoids_training(tmp_path):
    a, b, c = tmp_path / "a.jsonl", tmp_path / "b.jsonl", tmp_path / "c.jsonl"
    asyncio.run(H.build_heldout(seed=3, out=a, train_path=tmp_path / "yok.jsonl", plan=SMALL_PLAN))
    asyncio.run(H.build_heldout(seed=3, out=b, train_path=tmp_path / "yok.jsonl", plan=SMALL_PLAN))
    assert a.read_text() == b.read_text()
    rows = H.load_heldout(a)
    assert [r["cat"] for r in rows] == ["s1", "s2", "s4", "s5", "other", "error"]
    assert all(r["gold_calls"] and r["gold_action"] for r in rows)
    assert not rows[-1]["gold_calls"][0]["ok"] or rows[-1]["kind"] == "e_future"   # hata sorusu altını
    fake = tmp_path / "train.jsonl"                              # aynı sorular 'eğitimde' olsaydı
    fake.write_text("".join(json.dumps(t, ensure_ascii=False) + "\n" for t in _as_training(rows)))
    stats = asyncio.run(H.build_heldout(seed=3, out=c, train_path=fake, plan=SMALL_PLAN))
    old = {H.subject_key(r["cat"], r["asof"], r["gold_calls"]) for r in rows}
    new = {H.subject_key(r["cat"], r["asof"], r["gold_calls"]) for r in H.load_heldout(c)}
    assert not old & new and sum(stats["skipped"].values()) >= 1


# ---------------------------------------------------------------- puanlama (karar 3a)
def test_stated_dates_all_formats():
    q = "1 Ocak 2024 – 01.10.2025 arasında, 2023 başından bu yana; test 2026-01-01 sonrası."
    assert E.stated_dates(q, "2026-09-10") == {"2024-01-01", "2025-10-01", "2023-01-01", "2026-01-01", "2026-09-10"}


def test_call_matching_key_arguments():
    stated = {"2024-01-01", "2026-06-25"}
    gold = dict(name="rank_sector_relative_strength", args=dict(sector="kimya ve enerji", horizon=10, date="2026-06-25"),
                ok=True)
    call = lambda **a: dict(name=gold["name"], arguments=json.dumps(a), ok=True)
    members = TOOLS._peer_group("kimya ve enerji")[1]
    assert E.call_matches(gold, call(sector="Kimya ve Enerji", horizon=10, date="2026-06-25"), stated)
    assert E.call_matches(gold, call(tickers=list(reversed(members)), horizon="10", date="2026-06-25"), stated)
    assert not E.call_matches(gold, call(sector="kimya ve enerji", horizon=20, date="2026-06-25"), stated)
    assert not E.call_matches(gold, call(tickers=["KİMAYA"], horizon=10, date="2026-06-25"), stated)
    ev = dict(name="detect_reversal_events", ok=True,
              args=dict(ticker="ASELS", event="bb_lower_touch", start="2024-01-01", end="2026-06-25",
                        conditions=["rsi_dip+hacim_yuksek", "st_asagi+bulut_alti"]))
    rec = lambda **a: dict(name=ev["name"], arguments=json.dumps(a), ok=True)
    same = dict(ticker="asels", event="bb_lower_touch", start="2024-01-01", end="2026-06-25")
    assert E.call_matches(ev, rec(**same, conditions=["bulut_alti+st_asagi", "hacim_yuksek+rsi_dip"]), stated)
    assert not E.call_matches(ev, rec(**same), stated)                        # istenen koşullar yok
    assert not E.call_matches(ev, rec(**dict(same, start="2025-01-01"), conditions=ev["args"]["conditions"]), stated)
    ind = dict(name="calculate_indicators", ok=True, args=dict(ticker="ASELS", start="2026-06-11", end="2026-06-25"))
    assert E.call_matches(ind, dict(name="calculate_indicators", ok=True,      # soruda geçmeyen başlangıç serbest,
                                    arguments=json.dumps(dict(ticker="ASELS", start="2026-01-01", end="2026-06-25",
                                                              indicators=["rsi_14"]))), stated)   # fazla parametre olur
    assert not E.call_matches(ind, dict(name="calculate_indicators", ok=True, arguments="{bozuk"), stated)


def test_expected_failures_and_gold_matching_in_error_question():
    q = dict(hid="h1", cat="error", bad="XYZAB", question="XYZAB hissesi 01.01.2025 – 10.02.2025 ne yaptı?",
             asof="2025-02-10", gold_action="ANALYSIS_UNSAFE",
             gold_calls=[dict(name="get_stock_history", ok=False,
                              args=dict(ticker="XYZAB", start="2025-01-01", end="2025-02-10"))])
    recs = [dict(name="get_stock_history", ok=False, arguments='{"ticker": "xyzab", "start": "2025-01-01", '
                                                               '"end": "2025-02-10"}'),
            dict(name="check_data_quality", ok=False, arguments='{"ticker": "XYZAB"}'),
            dict(name="calculate_indicators", ok=False, arguments='{"ticker": "THYAO", "end": "2030-01-01"}')]
    r = dict(model="v1", run=1, action="ANALYSIS_UNSAFE", status="ok", retries=0, tool_calls=recs, first_numbers=0,
             first_unsupported=[], first_advice=[], final_advice=[], llm_calls=2, latency=1.0)
    s = E.score(q, r)
    assert (s["gold_hit"], s["expected_fail"], s["invalid"], s["correct"]) == (1, 2, 1, True)


def test_aggregate_and_reproducibility():
    base = dict(cat="s1", run=1, gold_action="WATCH", status="ok", retries=0, n_calls=2, invalid=1, expected_fail=0,
                gold_n=2, gold_hit=1, full_evidence=False, bt_n=1, bt_hit=0, first_numbers=10, first_fabricated=1,
                first_advice=0, final_advice=0, llm_calls=3, latency=10.0)
    rows = [dict(base, hid="a", model="v1", action="WATCH", correct=True),
            dict(base, hid="b", model="v1", action="HOLD", correct=False, invalid=0, gold_hit=2, full_evidence=True,
                 bt_hit=1, first_fabricated=0, latency=30.0)]
    a = E.aggregate(rows)
    assert (a["action_accuracy"], a["invalid_call_rate"], a["evidence_completeness"]) == (0.5, 0.25, 0.75)
    assert (a["backtest_call_correct"], a["fabricated_number_rate"], a["latency_median"]) == (0.5, 0.05, 20.0)
    call = [dict(name="x", arguments="{}")]
    raw = [dict(hid="a", model="v1", run=1, action="WATCH", tool_calls=call, answer="t"),
           dict(hid="a", model="v1", run=2, action="WATCH", tool_calls=call, answer="t2")]
    assert E.reproducibility(raw)["v1"] == dict(n=1, same_action=1.0, same_calls=1.0, same_answer=0.0)


def test_repeat_subset_cycles_categories():
    qs = [dict(hid=f"{c}{i}", cat=c) for c in E.CATS for i in range(3)]
    assert [q["cat"] for q in E.repeat_subset(qs, 9)] == list(E.CATS) + ["s1", "s2"]


def test_run_one_and_summary_without_server(tmp_path):
    q = dict(hid="h01", cat="other", kind="o_price", question="THYAO hissesi 2025-01-02 ile 2025-02-03 arasında "
             "hangi kapanış fiyatlarını gördü?", asof="2025-02-03", gold_action="HOLD", targeted=False, bad=None,
             gold_calls=[dict(name="get_stock_history", ok=True,
                              args=dict(ticker="THYAO", start="2025-01-02", end="2025-02-03"))])
    (tmp_path / "heldout.jsonl").write_text(json.dumps(q, ensure_ascii=False) + "\n")
    llm = ScriptLLM([{"tool_calls": [("get_stock_history", dict(ticker="THYAO", start="2025-01-02", end="2025-02-03"))]},
                     {"content": "- THYAO kapanışları araç çıktısında listelendi.\nANALİTİK DURUM: HOLD"}])
    rec = E.run_one(q, llm, "v1", 1, tmp_path / "question_log.sqlite")
    assert rec["action"] == "HOLD" and rec["status"] == "ok" and rec["first_unsupported"] == []
    E._append(tmp_path / "raw.jsonl", rec)
    E.replay_all(tmp_path)
    res = E.summarize(tmp_path)
    assert res["overall"]["v1"]["action_accuracy"] == 1.0 and res["overall"]["v1"]["full_evidence"] == 1.0
    assert res["replay_same_outputs"]["v1"] == 1.0
    assert "action accuracy" in (tmp_path / "summary.md").read_text()
    assert "THYAO hissesi" in (tmp_path / "per_question.csv").read_text()
