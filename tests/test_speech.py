from pathlib import Path

import pandas as pd
import pytest

from src.data.speech import SEGMENTS_FILE, SOURCES, load_speech_segments, match_rate, merge_to_sentences
from src.data.speech_features import (FEATURES_FILE, get_speech_features, load_speech_features, percents_en,
                                      percents_tr, stance, verify)


def test_percents_en_words_digits_and_ranges():
    assert percents_en("the federal funds rate at three and three quarters to four percent in") == [3.75, 4.0]
    assert percents_en("revised up from an earlier estimate of 1.8% up to 2.4%") == [1.8, 2.4]
    assert percents_en("growth of two point five percent") == [2.5]
    assert percents_en("about twenty-five percent of firms") == [25.0]
    assert percents_en("target rate at 3¾ to 4 percent") == [3.75, 4.0]          # resmi metindeki kesir
    assert percents_en("eight meetings per year and two for strategy") == []      # yüzde olmayan sayı alınmaz


def test_percents_tr():
    assert percents_tr("Yıllık enflasyon yüzde 29,8 oldu, faiz %36") == [29.8, 36.0]
    assert percents_tr("100 baz puan indirim") == []


def test_match_rate():
    assert match_rate("good day this morning", "Good day. This morning, the FOMC") == 1.0
    assert match_rate("good day wrong words", "good day this morning") == 0.5
    assert match_rate("", "anything") == 0.0


def test_merge_to_sentences_by_punctuation_and_word_cap():
    parts = pd.DataFrame(dict(start=[0.0, 2.0, 5.0, 9.0], end=[2.0, 5.0, 9.0, 12.0],
                              text=["Good day", "rates unchanged.", " ".join(["word"] * 45), "last part"],
                              avg_logprob=[-0.2, -0.4, -0.3, -0.1]))
    m = merge_to_sentences(parts)
    assert m["text"].tolist()[0] == "Good day rates unchanged."
    assert (m["start"].tolist(), m["end"].tolist()) == ([0.0, 5.0, 9.0], [5.0, 9.0, 12.0])   # 45 kelime ayrı kesilir


def test_verify_point_in_time():
    s = "Ekim ayında yıllık tüketici enflasyonu yüzde 29,8 olarak gerçekleşti."
    assert verify(s, [29.8], "2026-11-12")["status"] == "VERIFIED"
    assert verify(s, [29.8], "2026-11-02")["status"] != "VERIFIED"     # Ekim TÜFE'si 3 Kasım'da yayınlandı
    f = "the target rate for the federal funds rate at 3¾ to 4 percent"
    assert verify(f, [3.75, 4.0], "2026-10-28")["status"] == "VERIFIED"
    assert verify(f, [3.75, 4.0], "2026-09-01")["status"] != "VERIFIED"   # o gün aralık 3,50-3,75 idi


def test_verify_forecast_contradiction_and_no_tool():
    assert verify("2026 yıl sonu enflasyon tahminimizi yüzde 29,5 olarak güncelledik.", [29.5],
                  "2026-11-12")["status"] == "FORECAST_OR_TARGET"
    assert verify("Politika faizi yüzde 50 seviyesinde.", [50.0], "2026-11-12")["status"] == "CONTRADICTED"
    assert verify("Temel mallarda artış yüzde 11 ile sınırlı kaldı.", [11.0], "2026-11-12")["status"] == "NO_TOOL"


def test_stance_label():
    assert stance(3, 1) == (0.5, "sahin")
    assert stance(0, 0) == (0.0, "notr")
    assert stance(1, 3) == (-0.5, "guvercin")


@pytest.mark.skipif(not (Path(SEGMENTS_FILE).exists() and Path(FEATURES_FILE).exists()),
                    reason="önce: python -m src.data.speech ve python -m src.data.speech_features")
def test_real_speech_table():
    seg = load_speech_segments()
    assert set(seg["source_id"]) == set(SOURCES)
    assert seg[["media_url", "official_text_url", "speaker", "channel", "publish_time"]].notna().all().all()
    fed = seg[seg["source_id"] == "fed_20261028"]
    assert (fed["start"] < fed["end"]).all() and fed["start"].is_monotonic_increasing   # video zaman aralığı
    feats = load_speech_features()
    assert (feats.loc[feats["in_evidence"], "status"] == "VERIFIED").all()           # kanıta sadece doğrulanan
    r = get_speech_features(feats, "fed_20261028")
    assert r["match_rate"] >= 0.85 and r["warnings"] == []
    assert any(c["variable"] == "fed_faiz_araligi" and c["status"] == "VERIFIED" for c in r["claims"])
    t = get_speech_features(feats, "tcmb_20261112")
    assert any("YouTube" in w for w in t["warnings"])
    with pytest.raises(ValueError):
        get_speech_features(feats, "yok")
