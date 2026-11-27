"""v4 (27 Kasım 2026): hata-düzelt konuşmaları, otomatik S2 desen cümleleri, kısa dönem kur iddiası, yeni istem."""
import json
import random

import pytest

from src.finetune import dataset_v4 as D4
from src.finetune import heldout as H
from src.finetune import heldout_v3 as H3
from src.finetune import templates as TP
from src.finetune.masks import assistant_spans
from src.harness import question_mode as QM
from src.mcp_server import tools as TOOLS
from tests.test_finetune_masks import CONV, qwen_like


def test_week_codes_parse_and_cover_classic_patterns():
    """Bütün üretilen desenler aracın dilbilgisine uyar; öğretmenin ve v1-v3'ün 8 deseni aralarında."""
    for code in D4.WEEK_CODES:
        TOOLS._parse_week("week:" + code)
    assert set(TP.WEEK_PATTERNS) <= set(D4.WEEK_CODES) and len(D4.WEEK_CODES) >= 100
    assert D4.week_code(0, 1, "D", 2, 2, "U") == "DD-U.." and D4.week_code(0, 0, "D", 4, 4, "U") == "D...-U"


def test_heldout_v3_phrasings_not_in_v4_training():
    """Karar (27 Kasım 2026): heldout_v3 kalıpları ve desen ifadeleri v4 eğitiminde yok."""
    descs = {d for r in D4.WEEK_CODES.values() for d in D4.week_descriptions(*r)}
    assert not set(H3.WEEK_EVAL3.values()) & descs
    train_q = {t for k in D4.SamplerV4.Q for t in D4.SamplerV4.Q[k]} | set(D4.Q_V4_SHORT)
    assert not {t for k in H3.Q_EVAL3 for t in H3.Q_EVAL3[k]} & train_q
    smp = D4.SamplerV4(random.Random(0))
    assert not {smp.WEEK[c] for c in TP.WEEK_PATTERNS for _ in range(30)} & set(H3.WEEK_EVAL3.values())


def test_bad_args_break_the_call_and_tool_rejects_them():
    cases = [("strategy", dict(strategy="catch_up")), ("strategy", dict(strategy="A_teknik")),
             ("pattern", dict(pattern="week:DD-U..")), ("pattern", dict(pattern="up_streak:3")),
             ("event", dict(event="bb_lower_touch")), ("cond", dict(conditions=["rsi_dip+hacim_yuksek"])),
             ("ind", dict(indicators=["rsi_14", "st_dir"])), ("ticker", dict(ticker="THYAO")),
             ("ticker", dict(tickers=["BIMAS", "MGROS"])), ("index", dict(index="XU100")),
             ("swap", dict(start="2025-01-01", end="2025-03-01"))]
    for kind, args in cases:
        for u in (0.0, 0.5, 0.99):
            bad = D4.bad_args(kind, args, u)
            assert bad is not None and bad != args, (kind, args, u)
    assert D4.bad_args("cond", dict(conditions=None), 0.1) is None
    assert D4.bad_args("swap", dict(start="2025-03-01", end="2025-01-01"), 0.1) is None
    with pytest.raises(TOOLS.ToolError):
        TOOLS._ticker(D4.bad_args("ticker", dict(ticker="THYAO"), 0.0)["ticker"])
    with pytest.raises(TOOLS.ToolError):
        TOOLS._parse_week(D4.bad_args("pattern", dict(pattern="week:DD-U.."), 0.5)["pattern"])
    mk = D4.make_mistake("run_backtest", "strategy", 0.3)
    assert mk("get_stock_history", dict(ticker="THYAO")) is None and mk("run_backtest", dict(strategy="catch_up"))


def test_untrained_assistant_message_is_context_not_target():
    """'train': False işaretli asistan mesajı tokenlerde var ama hedef aralığı yok; işaret şablona gitmez."""
    wrong = dict(role="assistant", content=None, tool_calls=[dict(function=dict(name="rnak"))], train=False)
    conv = CONV[:2] + [wrong, dict(role="tool", content="HATA: bilinmeyen")] + CONV[2:]
    seen = []
    ids, spans = assistant_spans(conv, lambda ms, g: seen.append(ms) or qwen_like(ms, g))
    got = ["".join(ids[a:b]) for a, b in spans]
    assert got == ["<tool_call>rank</tool_call><|im_end|>\n", "cevap\nANALİTİK DURUM: WATCH<|im_end|>\n"]
    assert "<tool_call>rnak</tool_call>" in "".join(ids)
    assert all("train" not in m for ms in seen for m in ms)


def test_prompt_rules_v4():
    s = QM.SYSTEM
    assert "düzeltip yeniden çağır" in s and "2024-01-01" in s and "yalnızca bu üç eylemden biri" in s
    assert "ilişki hesaplanamadıysa iddia" in s and "düzeltilemeyen araç hatası" in s


def test_short_fx_claim_ends_investigate_and_passes_checks():
    """Kısa dönemde araç kur-hisse ilişkisini hesaplamaz; şablon INVESTIGATE der ve cevap denetimden geçer."""
    q = "Son haftalarda dolar yükseldi; MGROS 16.09.2024 – 18.10.2024 arasında kesin yükselir mi?"
    with TOOLS.data_ceiling("2024-10-18"):
        o = TOOLS.get_macro_context(start="2024-09-16", end="2024-10-18", ticker="MGROS")
    assert "daily" not in o["fx_relation"] and o["fx_relation"]["n_days"] < 60
    rec = dict(name="get_macro_context", ok=True, output=QM.compact("get_macro_context", TOOLS.to_json(o)),
               arguments="{}", args={})
    text, action = TP.ans_s5_fx(dict(T="MGROS"), [rec])
    assert action == "INVESTIGATE" and "hesaplandı mı? hayır" in text
    v = QM.validate_answer(text, q, [rec])
    assert not v["problems"] and v["action"] == "INVESTIGATE"


@pytest.fixture(scope="module")
def small_v4(tmp_path_factory):
    out = tmp_path_factory.mktemp("v4")
    D4.build_v4(dict(s2=3, s5=3, fix=6), out=out)
    convs = [json.loads(line) for line in open(out / "conversations.jsonl")]
    return convs, open(out / "train.jsonl").read() + open(out / "valid.jsonl").read()


def test_fix_conversations_break_once_then_correct(small_v4):
    convs, turns = small_v4
    fixes = [c for c in convs if c.get("fix")]
    assert len(fixes) == 6
    for c in fixes:
        ms = c["messages"]
        i = next(i for i, m in enumerate(ms) if m.get("train") is False)
        assert sum(m.get("train") is False for m in ms) == 1
        bad_ids = [tc["id"] for tc in ms[i]["tool_calls"]]
        bad = {tc["function"]["name"] for tc in ms[i]["tool_calls"]}
        errs = [m for m in ms[i + 1:] if m["role"] == "tool" and m["tool_call_id"] in bad_ids and m["content"].startswith("HATA")]
        assert len(errs) == 1, c["fix"]
        nxt = next(m for m in ms[i + 1:] if m["role"] == "assistant")
        assert nxt.get("train", True) and {tc["function"]["name"] for tc in nxt["tool_calls"]} <= bad
        assert c["action"] != "ANALYSIS_UNSAFE"                  # düzeltilebilir hata analizi durdurmaz
        gold = H.gold_calls(ms)                                  # bozuk çağrı altın sayılmaz
        assert all(g["ok"] for g in gold) and len(gold) == sum(len(m.get("tool_calls") or []) for m in ms
                                                               if m["role"] == "assistant" and m.get("train", True))
    assert '"train"' not in turns


def test_v4_heldout_phrasings_are_new():
    """heldout_v4 kalıpları, terimleri ve desen cümleleri v4 eğitiminde de heldout_v3'te de yok."""
    from src.finetune import heldout_v4 as H4
    train = {t for k in D4.SamplerV4.Q for t in D4.SamplerV4.Q[k]} | set(D4.Q_V4_SHORT)
    v3 = {t for k in H3.Q_EVAL3 for t in H3.Q_EVAL3[k]}
    new = [t for k in H4.Q_EVAL4 for t in H4.Q_EVAL4[k]]
    assert set(H4.Q_EVAL4) - {"s5_short"} == set(TP.Q) and not set(new) & (train | v3)
    for new_bank, olds in ((H4.EVENT_EVAL4, (TP.EVENT_TR, H.EVENT_EVAL, H3.EVENT_EVAL3)),
                           (H4.STRAT_EVAL4, (TP.STRAT_TR, H.STRAT_EVAL, H3.STRAT_EVAL3))):
        assert set(new_bank) == set(olds[0]) and not set(new_bank.values()) & {v for b in olds for v in b.values()}
    old_conds = {t for bank in (D4.SamplerV4.CONDS, H3.COND_EVAL3) for d in bank for t, _ in bank[d] if t}
    assert {t for d in H4.COND_EVAL4 for t, _ in H4.COND_EVAL4[d] if t}.isdisjoint(old_conds)
    train_desc = {d for r in D4.WEEK_CODES.values() for d in D4.week_descriptions(*r)} | \
                 set(TP.WEEK_PATTERNS.values()) | set(H.WEEK_EVAL.values()) | set(H3.WEEK_EVAL3.values())
    h_desc = {d for r in D4.WEEK_CODES.values() for d in H4.h_descriptions(*r)}
    assert h_desc and not h_desc & train_desc
    assert "MACD" in H4.Q_EVAL4["o_ind"][0] and "Bollinger" in H4.Q_EVAL4["o_ind"][1]
