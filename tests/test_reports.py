import numpy as np
import pandas as pd

from src.reports import common
from src.reports.common import bh, clustered_t, load_notes, md_table


def test_md_table_formats_and_missing_values():
    df = pd.DataFrame({"a": [1.234, np.nan], "n": [1200, 3], "ok": [True, False]}, index=["x", "y"])
    t = md_table(df, fmt={"a": "{:.1f}"}).splitlines()
    assert t[0] == "| index | a | n | ok |" and t[1] == "|:---|---:|---:|:---|"
    assert t[2] == "| x | 1.2 | 1,200 | yes |" and t[3] == "| y | – | 3 | no |"


def test_notes_are_split_by_key(tmp_path, monkeypatch):
    monkeypatch.setattr(common, "NOTES", tmp_path)
    (tmp_path / "r.md").write_text("# başlık\n<!-- summary -->\nilk\n\n<!-- main -->\nikinci\nsatır\n")
    n = load_notes("r.md")
    assert n == {"summary": "ilk", "main": "ikinci\nsatır"}
    assert "not written yet" in common.note(n, "yok")


def test_day_clustering_and_bh():
    dates = pd.Series(pd.to_datetime(["2024-01-02"] * 50 + ["2024-01-03"] * 50 + ["2024-01-04"] * 50))
    vals = pd.Series([1.0] * 50 + [-0.5] * 50 + [0.2] * 50)
    t, p, nd = clustered_t(vals, dates)
    assert nd == 3 and p > 0.3                       # 150 olay ama yalnızca 3 bağımsız gün
    q = bh([0.01, 0.04, np.nan, 0.5])
    assert np.isnan(q[2]) and q[0] <= q[1] <= q[3] and abs(q[0] - 0.03) < 1e-9


def test_report_1_builds_without_missing_notes():
    from src.reports.r1_technical import build
    text = open(build(), encoding="utf-8").read()
    assert "not written yet" not in text
    for s in ("## 1. Research question", "Table 2.", "Table 7.", "## 12. Limitations", "fig/01_regimes.png"):
        assert s in text


def test_report_2_builds_and_uses_point_in_time_earnings():
    from src.reports.r2_sector import build, load_events
    s = load_events()
    seen = s.dropna(subset=["event_day"])
    assert (seen["event_day"] <= seen["date"]).all()                 # bilanço olay gününden önce açıklanmış olmalı
    assert not s["quake_window"].any()
    text = open(build(), encoding="utf-8").read()
    assert "not written yet" not in text
    for k in ("Table 2. Filter ladder", "Table 6. Worst cases", "Table 8. Backtest", "fig/02_filter_ladder.png"):
        assert k in text


def test_md_table_keeps_integers_in_mixed_rows():
    df = pd.DataFrame({"trades": [7500], "ret": [0.123]})
    assert md_table(df, index=False).splitlines()[2] == "| 7,500 | 0.12 |"


def test_report_3_builds_with_single_multiple_testing_family():
    from src.reports.r3_weekday import build
    text = open(build(), encoding="utf-8").read()
    assert "not written yet" not in text
    for k in ("Table 1. Portfolio return by weekday", "Table 8. Wednesday-buy", "correcting all 180 tests",
              "fig/03_wed_thu_costs.png"):
        assert k in text


def test_report_4_builds_with_clean_tables():
    from src.reports.r4_earnings import build
    text = open(build(), encoding="utf-8").read()
    assert "not written yet" not in text
    for k in ("Table 2. Top-minus-bottom", "Table 5. Top-minus-bottom third spread by metric", "Table 8. Net-income third",
              "fig/04_car_path.png"):
        assert k in text
    assert "|AR|" not in text                                     # boru karakteri tabloyu bozmamalı


def test_report_5_builds_and_only_verified_claims_enter_evidence():
    import pandas as pd
    from src.reports.r5_macro_context import build
    f = pd.read_parquet("reports/speech_features.parquet")
    assert (f.loc[f["in_evidence"], "status"] == "VERIFIED").all()
    text = open(build(), encoding="utf-8").read()
    assert "not written yet" not in text
    for k in ("Table 2. Year by year", "Table 5. Sector FX sensitivity", "Table 13. Claims that entered",
              "fig/05_macro_timeline.png"):
        assert k in text


def test_data_dictionary_describes_every_column():
    """Bir veri setine yeni sütun eklenir ya da biri silinirse sözlük güncellenmeden bu test geçmez."""
    from src.reports.data_dictionary import DATASETS, check
    problems = {ds["key"]: check(ds) for ds in DATASETS}
    assert all(not m and not x for m, x in problems.values()), {k: v for k, v in problems.items() if v[0] or v[1]}


def test_report_6_builds_and_replays_real_log():
    from src.reports.r6_system import build, lookahead_check
    la = lookahead_check()
    assert la.iloc[0]["total %"] > 1e6 > la.iloc[1]["total %"]          # kaydırma yoksa absürt, kaydırınca makul
    text = open(build(), encoding="utf-8").read()
    assert "not written yet" not in text
    for k in ("Table 1. Strategy results", "Table 8. Action per case", "Table 13. Replay", "fig/06_experiments.png"):
        assert k in text


def test_final_report_builds_and_pdf_keeps_nested_lists():
    from src.reports.final_report import build
    from src.reports.pdf import to_html
    path = build()
    text = open(path, encoding="utf-8").read()
    assert "not written yet" not in text and "fig/final_architecture.png" in text
    for k in ("## 9. Seven distinctions", "Table 4. Six cases", "Appendix A. Deliverables"):
        assert k in text
    html = to_html([path])
    assert "<ul>\n<li>" in html and html.count("<ul>") > html.count("<h2")      # iç içe listeler korunur
    assert "<pre><code" in html                                                # ``` blokları kod olarak
