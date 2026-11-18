import pandas as pd

from src.events.news_reaction import classify, news_flags, news_signal_events


def test_classify_rules():
    assert classify("Özel Durum Açıklaması (Genel) Genel Müdür Ayrılışı") == ("olumsuz", "yonetici_ayrilis")
    assert classify("Finansal Sonuçların Açıklanmasının Ertelenmesi Hk.") == ("olumsuz", "bilanco_erteleme")
    assert classify("Yeni İş İlişkisi ASELSAN sözleşme imzaladı") == ("olumlu", "yeni_is_sozlesme")
    assert classify("Yeni İş İlişkisi; dava açıldı") == ("olumsuz", "dava_ceza")      # ikisi birden: olumsuz öncelikli
    assert classify("Genel Kurul İşlemlerine İlişkin Bildirim") == (None, None)


def test_news_flags_window_and_no_lookahead():
    idx = pd.bdate_range("2026-10-01", periods=15)
    ev = pd.DataFrame(dict(ticker=["A", "B"], event_day=[idx[3], idx[5]], direction=["olumsuz", "olumlu"]))
    neg, pos = news_flags(idx, ["A", "B"], ev, window=3)
    assert neg["A"].tolist()[:7] == [False, False, False, True, True, True, False]   # olay gününden önce görünmez
    assert not neg["B"].any() and pos["B"].sum() == 3 and not pos["A"].any()


def test_real_signal_events_use_event_day():
    ev = news_signal_events()
    assert set(ev["direction"]) == {"olumsuz", "olumlu"}
    assert (ev["event_day"] >= ev["publish_time"].dt.normalize()).all()             # haber geçmişe kaymaz
    assert not ev.duplicated(["ticker", "event_day", "kind"]).any()
