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
