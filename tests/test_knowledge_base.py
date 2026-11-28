"""RAG dosyası (src/submission/knowledge_base.py): bilgi tabanı belgeleri + kaynak meta verileri."""
import hashlib
import json

import pandas as pd
import pytest

from src.harness import question_mode as QM
from src.submission import knowledge_base as K


@pytest.fixture(scope="module")
def kb(tmp_path_factory):
    kb = K.build(tmp_path_factory.mktemp("x") / "yok.pdf")          # PDF yoksa da çalışır (yalnızca künye)
    json.dumps(kb, ensure_ascii=False, allow_nan=False)            # geçerli JSON: NaN/NaT yok
    return kb


def test_groups_match_sources(kb):
    c = {g["group"]: g["count"] for g in kb["groups"]}
    ne = pd.read_parquet(K.NEWS)
    assert c["kap_disclosure"] == (ne.source == "KAP").sum() and c["rss_headline"] == (ne.source != "KAP").sum()
    assert c["speech_segment"] == len(pd.read_parquet(K.SPEECH)) and c["speech_source"] == 2
    assert c["action_rule"] == 7 and c["action_definition"] == len(QM.ACTION_HELP) == 8 and c["tool_definition"] == 15
    assert c["data_dictionary"] == 23 and kb["n_documents"] == sum(c.values()) == len(kb["documents"])


def test_every_document_has_common_fields_and_hash(kb):
    keys = set(kb["fields"])
    ids = set()
    for d in kb["documents"]:
        assert set(d) == keys and d["text"] and d["usage"] in kb["usage_notes"]
        assert d["sha256"] == hashlib.sha256(d["text"].encode("utf-8")).hexdigest()
        assert d["doc_id"] not in ids
        ids.add(d["doc_id"])


def test_news_and_speech_carry_source_metadata(kb):
    for d in kb["documents"]:
        if d["group"] in ("kap_disclosure", "rss_headline", "speech_segment", "speech_claim", "speech_source"):
            assert d["url"].startswith("https://") and d["publish_time"] and d["publisher"]
        if d["group"] == "kap_disclosure":
            assert d["fetched_at"] and d["ticker"] and d["doc_id"] == f"kap_{d['extra']['kap_index']}_{d['ticker']}"
        if d["group"] == "speech_segment" and d["extra"]["source_id"].startswith("fed_"):     # ses: Whisper saniyeleri
            assert d["extra"]["start_seconds"] < d["extra"]["end_seconds"]
        if d["group"] == "speech_segment" and d["extra"]["source_id"].startswith("tcmb_"):    # resmi metin: saniye yok
            assert d["extra"]["start_seconds"] is None


def test_rules_are_the_live_system_prompt(kb):
    rules = [d["text"] for d in kb["documents"] if d["group"] == "action_rule"]
    flat = " ".join(QM.SYSTEM.split())
    assert all(r in flat for r in rules) and "ANALYSIS_UNSAFE" in rules[-1]


def test_structured_sources_have_file_hashes(kb):
    by = {s["source_id"]: s for s in kb["structured_sources"]}
    assert {"prices_isyatirim", "fundamentals_fintables", "kap_financial_reports", "macro_evds", "fed_funds",
            "project_spec"} <= set(by)
    assert len(by["prices_isyatirim"]["files"]) == 34 and len(by["fundamentals_fintables"]["files"]) == 30
    assert all(len(f["sha256"]) == 64 for s in by.values() for f in s.get("files", []))
    assert "pages" not in by["project_spec"]                       # PDF verilmediyse künye uydurulmaz
