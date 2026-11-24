"""Yerel model 4. adım: konuşma başına kayıp maskesi (src/finetune/masks.py). Gerçek tokenizer yerine Qwen biçimini
taklit eden karakter düzeyinde bir şablon kullanılır (her karakter bir token)."""
import pytest

from src.finetune.masks import assistant_spans, target_mask


def qwen_like(messages, add_generation_prompt):
    out = ""
    for m in messages:
        body = m.get("content") or ""
        if m.get("tool_calls"):
            body += "".join(f"<tool_call>{c['function']['name']}</tool_call>" for c in m["tool_calls"])
        out += f"<|im_start|>{m['role']}\n{body}<|im_end|>\n"
    if add_generation_prompt:
        out += "<|im_start|>assistant\n"
    return list(out)


CONV = [dict(role="system", content="kurallar"), dict(role="user", content="soru"),
        dict(role="assistant", content=None, tool_calls=[dict(function=dict(name="rank"))]),
        dict(role="tool", content='{"x": 1}'),
        dict(role="assistant", content="cevap\nANALİTİK DURUM: WATCH")]


def test_spans_cover_exactly_the_assistant_turns():
    ids, spans = assistant_spans(CONV, qwen_like)
    text = "".join(ids)
    got = ["".join(ids[a:b]) for a, b in spans]
    assert got == ["<tool_call>rank</tool_call><|im_end|>\n", "cevap\nANALİTİK DURUM: WATCH<|im_end|>\n"]
    assert len(spans) == 2 and text.startswith("<|im_start|>system\nkurallar")


def test_target_mask_learns_only_assistant_tokens():
    ids, spans = assistant_spans(CONV, qwen_like)
    m = target_mask(len(ids), spans)
    learned = "".join(ids[k + 1] for k, v in enumerate(m) if v)          # hedef = bir sonraki token
    assert learned == "<tool_call>rank</tool_call><|im_end|>\ncevap\nANALİTİK DURUM: WATCH<|im_end|>\n"
    assert "soru" not in learned and '{"x": 1}' not in learned and "kurallar" not in learned
    assert len(m) == len(ids) - 1


def test_non_prefix_consistent_template_is_rejected():
    def bad(messages, add_generation_prompt):                           # son asistan mesajını farklı yazan şablon
        out = qwen_like(messages, add_generation_prompt)
        return out + list("<son>") if messages and messages[-1]["role"] == "assistant" else out
    with pytest.raises(ValueError, match="önek tutarlı"):
        assistant_spans(CONV, bad)
