"""Yerel model 3. adım: ince ayar veri seti (src/finetune/build_dataset.py). Küçük ölçekli üretimle denenir."""
import asyncio
import hashlib
import json
import re

import pytest

from src.agents.llm import ScriptLLM
from src.finetune import build_dataset as B
from src.harness import question_mode as QM

SMALL = dict(s1=3, s2=2, s3=2, s4=2, s5=2, other=3, error=2)


@pytest.fixture(scope="module")
def small(tmp_path_factory):
    out = tmp_path_factory.mktemp("ft")
    stats = asyncio.run(B.build(SMALL, seed=7, out=out))
    convs = [json.loads(l) for l in open(out / "conversations.jsonl")]
    return out, stats, convs


def test_counts_teacher_questions_and_splits(small):
    out, stats, convs = small
    assert stats["conversations"] == sum(SMALL.values()) + 5 and stats["teacher"] == 5
    teacher = [c for c in convs if c["teacher"]]
    assert {c["question"] for c in teacher} == {t["question"] for t in B.TEACHER}
    assert all(c["split"] == "train" for c in teacher)                         # karar 3b: hocanın soruları eğitimde
    assert [c["action"] for c in teacher] == ["WATCH", "REJECT_SIGNAL", "WATCH", "FUNDAMENTAL_IMPROVEMENT_CANDIDATE",
                                              "INVESTIGATE"]


def test_turn_examples_follow_mlx_chat_format(small):
    out, stats, convs = small
    rows = [json.loads(l) for name in ("train", "valid") for l in open(out / f"{name}.jsonl")]
    n_assistant = sum(m["role"] == "assistant" for c in convs for m in c["messages"])
    assert len(rows) == n_assistant == sum(stats["turn_examples"].values())
    for r in rows:
        assert r["messages"][-1]["role"] == "assistant" and len(r["tools"]) == 15   # son mesaj hedef (mask_prompt)
        assert r["messages"][0]["role"] == "system" and r["messages"][1]["role"] == "user"


def test_same_prompt_as_question_mode_and_nothing_after_asof(small):
    _, _, convs = small
    for c in convs:
        assert c["messages"][:2] == QM.first_messages(c["question"], c["asof"])
        for m in c["messages"]:
            if m["role"] == "tool" and not m["content"].startswith("HATA"):
                dates = re.findall(r"\d{4}-\d{2}-\d{2}", m["content"])
                assert max(dates) <= c["asof"], (c["cat"], c["question"][:60])
        assert QM.parse_action(c["messages"][-1]["content"]) == c["action"]


def test_each_conversation_replays_exactly_in_question_mode(small):
    """Model veri setindeki turları yazsaydı soru modu aynı araç çıktılarını, aynı denetimi ve aynı eylemi üretirdi."""
    _, _, convs = small
    for c in convs:
        steps = []
        for m in c["messages"][2:]:
            if m["role"] == "assistant":
                steps.append({"tool_calls": [(tc["function"]["name"], json.loads(tc["function"]["arguments"]))
                                             for tc in m["tool_calls"]]} if m.get("tool_calls") else {"content": m["content"]})
        res = QM.run_question(c["question"], c["asof"], ScriptLLM(steps), log=False)
        tool_msgs = [m["content"] for m in c["messages"] if m["role"] == "tool"]
        seen = [json.dumps(o["output"], ensure_ascii=False) if o["ok"] else f"HATA: {o['error']}" for o in res["tool_calls"]]
        assert seen == tool_msgs, c["question"][:80]
        assert res["retries"] == 0 and res["validation"]["problems"] == []
        expect = "ANALYSIS_UNSAFE" if c["cat"] == "error" else c["action"]
        assert res["action"] == expect and res["answer"] == c["messages"][-1]["content"]


def test_catch_up_action_rule_rederived(small):
    """S1 kuralı şablondan bağımsız yeniden hesaplanır (sektör güçlü, >= 10 olay, isabet >= %60, YP <= %40)."""
    _, _, convs = small
    for c in [c for c in convs if c["cat"] == "s1" and c.get("variant") != "rank"]:
        outs = [json.loads(m["content"]) for m in c["messages"] if m["role"] == "tool"]
        sm = outs[0]["sector_median"]
        st = outs[1]["stock"]
        h20 = next(x for x in st["by_horizon"] if x["horizon_days"] == 20)
        ok = sm > 0 and h20["n"] >= 10 and (h20["hit_rate"] or 0) >= 0.6 and (st["false_positive_rate_20"] or 1) <= 0.4
        assert c["action"] == ("POTENTIAL_CATCH_UP_CANDIDATE" if ok else "INVESTIGATE" if h20["n"] < 10 else "WATCH")


def test_build_is_reproducible(small, tmp_path):
    out, _, _ = small
    asyncio.run(B.build(SMALL, seed=7, out=tmp_path))
    sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
    assert sha(out / "conversations.jsonl") == sha(tmp_path / "conversations.jsonl")
    assert sha(out / "train.jsonl") == sha(tmp_path / "train.jsonl")
