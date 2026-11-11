
import pandas as pd

from src.data.macro import (_asof, cpi_release_date, fed_table, get_macro_context, load_series,
                            macro_panel, year_chunks)


def test_year_chunks():
    c = year_chunks("2020-08-01", "2022-03-15")
    assert c == [(pd.Timestamp("2020-08-01"), pd.Timestamp("2020-12-31")),
                 (pd.Timestamp("2021-01-01"), pd.Timestamp("2021-12-31")),
                 (pd.Timestamp("2022-01-01"), pd.Timestamp("2022-03-15"))]
    assert all((b - a).days < 1000 for a, b in c)                     # EVDS 1000 satır sınırı


def test_cpi_release_rule():
    assert cpi_release_date(pd.Timestamp("2026-10-01")) == pd.Timestamp("2026-11-03")
    assert cpi_release_date(pd.Timestamp("2026-09-01")) == pd.Timestamp("2026-10-05")   # 3 Ekim Cumartesi
    assert cpi_release_date(pd.Timestamp("2025-12-01")) == pd.Timestamp("2026-01-05")   # 3 Ocak Cumartesi


def test_asof_uses_only_published_values():
    df = pd.DataFrame({"available_date": pd.to_datetime(["2024-01-03", "2024-02-05"]), "value": [1.0, 2.0]})
    days = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-02-02", "2024-02-05", "2024-02-06"])
    assert _asof(df, days).isna().tolist() == [True, False, False, False, False]
    assert _asof(df, days).tolist()[1:] == [1.0, 1.0, 2.0, 2.0]


def test_cpi_not_visible_before_release():
    p = macro_panel()
    assert p.loc["2026-11-02", "cpi_month"] == "2026-09"              # Ekim TÜFE'si 3 Kasım'da yayınlandı
    assert p.loc["2026-11-03", "cpi_month"] == "2026-10"
    assert p.loc["2022-12-30", "cpi_month"] == "2022-11"
    assert abs(p.loc["2022-12-30", "cpi_yoy"] - 84.39) < 0.05         # Kasım 2022 yıllık TÜFE (TÜİK: %84,39)


def test_policy_rate_known_decisions():
    p = macro_panel()
    assert p.loc["2023-06-21", "policy_rate"] == 8.5 and p.loc["2023-06-22", "policy_rate"] == 15.0
    assert p.loc["2024-03-20", "policy_rate"] == 45.0 and p.loc["2024-03-21", "policy_rate"] == 50.0
    assert p.loc["2025-03-24", "policy_rate"] == 42.5                 # fonlama maliyeti %46'ya çıktı, politika faizi değişmedi
    assert p.loc["2023-06-22", "rate_change"] == 6.5
    r = load_series("policy_rate")
    ch = r[r["value"] != r["value"].shift()].iloc[1:]
    assert (ch["data_date"].dt.day_of_week == 3).all()                # PPK kararları Perşembe


def test_fed_table():
    f = fed_table()
    assert f["data_date"].is_monotonic_increasing and ((f["fed_upper"] - f["fed_lower"]).round(2) == 0.25).all()
    p = macro_panel()
    assert p.loc["2022-03-16", "fed_upper"] == 0.25 and p.loc["2022-03-17", "fed_upper"] == 0.50
    assert p.loc["2023-12-29", "fed_upper"] == 5.50 and p.loc["2026-09-17", "fed_upper"] == 4.00


def test_panel_complete_and_context():
    p = macro_panel()
    assert len(p) == 1258
    assert not p[["usdtry", "eurtry", "policy_rate", "cpi_yoy", "fed_lower", "fed_upper"]].isna().any().any()
    one = get_macro_context("2024-06-03")
    assert len(one) == 1 and one["usdtry"].iloc[0] > 30
    assert len(get_macro_context("2024-06-03", "2024-06-07")) == 5
