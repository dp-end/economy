"""v3 (26 Kasım 2026): kural satırlı eğitim verisi, karışık ifade havuzu ve yeni görülmemiş set."""
import json
import random

import pytest

from src.finetune import build_dataset as B
from src.finetune import dataset_v3 as D3
from src.finetune import heldout as H
from src.finetune import heldout_v3 as H3
from src.finetune import templates as TP


def test_v3_heldout_phrasings_are_new():
    """Yeni setin kalıpları ve terimleri ne eğitim kalıplarında ne de ilk görülmemiş sette (artık v3 eğitiminde)."""
    old = {t for k in TP.Q for t in TP.Q[k]} | {t for k in H.Q_EVAL for t in H.Q_EVAL[k]}
    new = [t for k in H3.Q_EVAL3 for t in H3.Q_EVAL3[k]]
    assert set(H3.Q_EVAL3) <= set(TP.Q) and not set(new) & old
    for new_bank, old_banks in ((H3.WEEK_EVAL3, (TP.WEEK_PATTERNS, H.WEEK_EVAL)),
                                (H3.EVENT_EVAL3, (TP.EVENT_TR, H.EVENT_EVAL)),
                                (H3.STRAT_EVAL3, (TP.STRAT_TR, H.STRAT_EVAL))):
        assert set(new_bank) == set(old_banks[0])
        assert not set(new_bank.values()) & {v for b in old_banks for v in b.values()}
    old_conds = {txt for d in B.COND_SETS for txt, _ in B.COND_SETS[d]} | {txt for d in H.COND_EVAL for txt, _ in H.COND_EVAL[d]}
    assert {txt for d in H3.COND_EVAL3 for txt, _ in H3.COND_EVAL3[d] if txt} .isdisjoint(old_conds)
    # o_ind: çift sıradaki kalıp RSI/MACD/Supertrend, tek sıradaki Bollinger (Sampler.other bu sıraya bakar)
    assert "MACD" in H3.Q_EVAL3["o_ind"][0] and "Bollinger" in H3.Q_EVAL3["o_ind"][1]
    assert H3.slash("2026-09-10") == "10/09/2026"


def test_v3_training_bank_mixes_old_and_first_heldout_phrasings():
    rng = random.Random(0)
    smp = D3.SamplerV3(rng)
    assert all(set(smp.Q[k]) == set(TP.Q[k]) | set(H.Q_EVAL[k]) for k in TP.Q)
    seen = {smp.WEEK["DDD-UU"] for _ in range(50)}
    assert seen == {TP.WEEK_PATTERNS["DDD-UU"], H.WEEK_EVAL["DDD-UU"]}
    assert sum(TP.Q["o_ind"].index(t) % 2 == i % 2 for i, t in enumerate(smp.Q["o_ind"]) if t in TP.Q["o_ind"]) == 2


@pytest.fixture(scope="module")
def small_v3(tmp_path_factory):
    out = tmp_path_factory.mktemp("v3")
    counts = dict(s1=2, s2=2, s3=2, s4=2, s5=2, other=1, error=1, s1_cand=1, s3_cand=1, s4_cand=1)
    D3.build_v3(counts, out=out)
    return [json.loads(line) for line in open(out / "conversations.jsonl")]


def test_v3_answers_have_rule_check_matching_action(small_v3):
    """Senaryo cevaplarında eylemden önce kural satırı var ve okun gösterdiği eylem son satırdakiyle aynı."""
    for c in small_v3:
        text = c["messages"][-1]["content"]
        lines = text.splitlines()
        if c["cat"] in ("s1", "s2", "s3", "s4") or (c["cat"] == "s5"):
            rule = next(l for l in lines if l.startswith("- Kural kontrolü: "))
            assert rule.endswith(f"→ {c['action']}") and lines.index(rule) == len(lines) - 3, text[-400:]
    t = {c["cat"]: c["action"] for c in small_v3 if c.get("targeted")}
    assert t == {"s1": "POTENTIAL_CATCH_UP_CANDIDATE", "s3": "TECHNICAL_REVERSAL_CANDIDATE",
                 "s4": "FUNDAMENTAL_IMPROVEMENT_CANDIDATE"}
