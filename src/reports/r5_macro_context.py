"""Rapor 5 — Macro and contextual evidence integration report (PDF Bölüm 15, 16, 25).

Kaynak: reports/macro_panel.csv (Adım 8, yayın tarihine hizalı), news_events.parquet / news_signal_events.parquet /
news_reaction_summary.csv (Adım 10), speech_features.parquet / speech_claims.csv (Adım 10B),
backtest_strategies.csv (Adım 9-10). Ek hesaplar: yıllık TRY/USD getirisi, makro risk bayrağı sonrası XU100,
sektörlerin kur duyarlılığı.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.backtest.strategies import FX_LOOKBACK, FX_PCT, FX_WINDOW, TREND_SMA, macro_scale
from src.data.macro import CPI_RELEASE_DAY, EVDS_SERIES
from src.data.prices import get_index_history, get_stock_history
from src.data.speech import load_speech_segments, quality
from src.data.universe import build_sector_peer_groups
from src.events.calendar_patterns import welch
from src.reports.common import C, DISCLAIMER, bh, load_notes, md_table, note, save_fig, stamp, style, write

SOURCES = ["reports/macro_panel.csv", "reports/news_events.parquet", "reports/news_signal_events.parquet",
           "reports/news_reaction_summary.csv", "reports/speech_features.parquet", "reports/speech_claims.csv",
           "reports/backtest_strategies.csv"]
NOTES_FILE = "05_macro_context.md"
MARKS = [("2021-12-20", "FX crisis /\nFX-protected deposits"), ("2023-06-22", "tightening\nstarts"),
         ("2024-12-26", "first cut"), ("2025-03-19", "political\nshock"), ("2026-09-17", "fund\nliquidations")]
CAT_EN = {"borclanma": "debt issuance", "diger": "other", "varant": "warrants", "geri_alim": "share buy-back",
          "sozlesme": "contracts / orders", "rutin": "routine filings", "yonetim": "management / board",
          "pay_alim_satim": "insider / major-holder trades", "genel_kurul": "general meeting",
          "derecelendirme": "credit rating", "birlesme_satinalma": "merger / acquisition", "temettu": "dividend"}
KIND_EN = {"geri_alim_baslangic": "buy-back programme started", "yeni_is_sozlesme": "new contract / order",
           "bilanco_erteleme": "financial statements delayed", "dava_ceza": "lawsuit / penalty / investigation",
           "tasfiye_iflas": "liquidation / bankruptcy", "yonetici_ayrilis": "executive departure"}
STATUS_EN = {"VERIFIED": "verified by a data tool", "CONSISTENT_WITH_HISTORY": "consistent with history",
             "FORECAST_OR_TARGET": "forecast or target (cannot be verified)", "NO_TOOL": "no data tool for this claim",
             "CONTRADICTED": "contradicted by data"}


def macro() -> pd.DataFrame:
    return pd.read_csv("reports/macro_panel.csv", parse_dates=["date"]).set_index("date")


def sources_table() -> pd.DataFrame:
    rows = [("USD/TRY, EUR/TRY", "TCMB EVDS " + EVDS_SERIES["usdtry"][0] + ", " + EVDS_SERIES["eurtry"][0], "daily",
             "known at that day's close"),
            ("TCMB policy rate (1-week repo)", "TCMB EVDS " + EVDS_SERIES["policy_rate"][0], "daily",
             "decision day (Thursday 14:00)"),
            ("CPI (TÜİK), yearly and monthly change", "TCMB EVDS " + EVDS_SERIES["cpi"][0], "monthly",
             f"{CPI_RELEASE_DAY}rd of the next month (business-day rule), never the reference month"),
            ("Fed funds target range", "federalreserve.gov, hand-entered table", "per decision",
             "first BIST day after the announcement (announced after BIST close)"),
            ("Real policy rate", "policy rate − yearly CPI", "daily", "from the two inputs above"),
            ("Macro risk flag", f"XU100 < {TREND_SMA}-day average, or 20-day USD/TRY rise in the top "
                                f"{round((1 - FX_PCT) * 100)}% of its last {FX_LOOKBACK} days", "daily",
             "threshold uses only yesterday's history")]
    return pd.DataFrame(rows, columns=["variable", "source", "frequency", "available on (point-in-time rule)"])


def yearly_table(m: pd.DataFrame) -> pd.DataFrame:
    xu = get_index_history("XU100")["Close"]
    y = pd.DataFrame({"xu": xu, "usd": m["usdtry"].reindex(xu.index).ffill()})
    ends = pd.concat([y.iloc[[0]], y.resample("YE").last()])
    r = ends.pct_change().dropna()
    last = m.resample("YE").last()
    out = pd.DataFrame({
        "XU100 in TRY %": r["xu"] * 100, "USD/TRY %": r["usd"] * 100,
        "XU100 in USD %": ((1 + r["xu"]) / (1 + r["usd"]) - 1) * 100,
        "CPI yearly at year end %": last["cpi_yoy"], "XU100 real (CPI-deflated) %":
            ((1 + r["xu"]) / (1 + last["cpi_yoy"] / 100) - 1) * 100,
        "policy rate at year end %": last["policy_rate"], "real policy rate at year end": last["real_rate"],
        "Fed upper bound at year end %": last["fed_upper"]})
    first, end = xu.index[0], xu.index[-1]
    out.index = [f"{d.year}" + (f" (from {first:%d %b})" if d.year == first.year else "")
                 + (f" (to {end:%d %b})" if d.year == end.year else "") for d in out.index]
    real_note = "CPI-deflated uses year-end yearly CPI; for part-years it overstates inflation"
    return out.rename_axis("year"), real_note


def risk_flag(m: pd.DataFrame) -> tuple:
    xu = get_index_history("XU100")["Close"]
    ms = macro_scale(xu.index)
    on = ms < 1
    rows = []
    for h in (5, 20):
        f = (xu.shift(-h) / xu - 1) * 100
        a, b = f[on].dropna(), f[~on].dropna()
        t, p = welch(a, b)
        rows.append({"horizon": f"next {h} days", "days flag on": len(a), "XU100 mean if on %": a.mean(),
                     "days flag off": len(b), "XU100 mean if off %": b.mean(), "t": t, "p": p})
    bt = pd.read_csv("reports/backtest_strategies.csv")
    keep = ["C_temel", "D_makro", "D_makro_oynaklik", "D_makro_faiz", "E_haber"]
    bt = bt[bt["strateji"].isin(keep)].copy()
    bt["XU100"] = bt["toplam_getiri"] - bt["xu100_fark"]
    tab = bt.rename(columns={"strateji": "strategy", "donem": "period", "toplam_getiri": "total return %",
                             "sharpe": "Sharpe", "max_dusus": "max drawdown %", "kiyas_getiri": "equal-weight B&H %",
                             "islem_sayisi": "trades", "XU100": "XU100 %"})
    tab["period"] = tab["period"].map({"tum": "full", "ilk_yari": "first half", "ikinci_yari": "second half"})
    for c in ("total return %", "max drawdown %", "equal-weight B&H %", "XU100 %"):
        tab[c] = tab[c] * 100
    tab = tab[["strategy", "period", "total return %", "XU100 %", "equal-weight B&H %", "Sharpe", "max drawdown %",
               "trades"]]
    tab["strategy"] = pd.Categorical(tab["strategy"], keep)
    return pd.DataFrame(rows).set_index("horizon"), tab.sort_values(["strategy"]).set_index(["strategy", "period"]), on.mean()


def fx_betas(m: pd.DataFrame) -> pd.DataFrame:
    groups, _ = build_sector_peer_groups()
    xu = get_index_history("XU100")["Close"]
    du = m["usdtry"].reindex(xu.index).pct_change() * 100
    xr = xu.pct_change() * 100
    rows = []
    for g, mem in groups.items():
        gr = pd.DataFrame({t: get_stock_history(t)["Close"].pct_change() * 100 for t in mem}).reindex(xu.index)
        ex = gr.mean(axis=1) - xr
        ok = ex.notna() & du.notna()
        res = sm.OLS(ex[ok], sm.add_constant(du[ok])).fit(cov_type="HAC", cov_kwds={"maxlags": 5})
        rows.append({"peer group": g, "stocks": ", ".join(mem), "beta to USD/TRY": res.params.iloc[1],
                     "t (HAC)": res.tvalues.iloc[1], "p": res.pvalues.iloc[1], "days": int(ok.sum())})
    out = pd.DataFrame(rows).set_index("peer group").sort_values("beta to USD/TRY")
    out["q BH"] = bh(out["p"])
    return out


def news_tables() -> tuple:
    n = pd.read_parquet("reports/news_events.parquet")
    src = n.groupby("source").agg(items=("title", "size"), first=("publish_time", "min"), last=("publish_time", "max"))
    src["first"] = pd.to_datetime(src["first"]).dt.date.astype(str)
    src["last"] = pd.to_datetime(src["last"]).dt.date.astype(str)
    cats = n[n["source"] == "KAP"]["category"].value_counts().head(12).rename("KAP disclosures").rename_axis("category")
    cats.index = [f"{CAT_EN.get(c, c)} (`{c}`)" for c in cats.index]
    cats = cats.rename_axis("category")
    scope = n.groupby(["source", "scope"]).size().unstack(fill_value=0)
    scope = scope.rename(columns={"sirket": "company", "makro": "macro", "genel": "general", "sektor": "sector"})
    s = pd.read_csv("reports/news_reaction_summary.csv")
    s = s.rename(columns={"direction": "direction", "kind": "kind", "n": "events", "hisse": "stocks",
                          "pre5_mean": "5 days before %", "d0_mean": "event day %", "d0_vs_base": "event day vs base",
                          "d0_p": "p (event day)", "d0_q": "q (event day)", "d0_4_mean": "days 0–4 %",
                          "post1_5_mean": "days 1–5 %", "az_olay": "too few events"})
    s["direction"] = s["direction"].map({"olumlu": "positive", "olumsuz": "negative"})
    s["kind"] = s["kind"].map(lambda k: KIND_EN.get(k, k))
    s = s[["direction", "kind", "events", "stocks", "5 days before %", "event day %", "event day vs base",
           "p (event day)", "q (event day)", "days 0–4 %", "days 1–5 %", "too few events"]]
    ev = pd.read_parquet("reports/news_signal_events.parquet")
    top = (ev[ev["kind"] == "yeni_is_sozlesme"]["ticker"].value_counts().head(5)
           .rename("contract events").rename_axis("ticker"))
    return src, cats, scope, s.set_index(["direction", "kind"]), top


def speech_tables() -> tuple:
    f = pd.read_parquet("reports/speech_features.parquet")
    seg = load_speech_segments()
    meta = f.groupby("source_id").agg(speaker=("speaker", "first"), channel=("channel", "first"),
                                      title=("title", "first"), published=("publish_time", "first"),
                                      language=("language", "first"), segments=("text", "size"),
                                      hawkish=("hawkish", "sum"), dovish=("dovish", "sum"),
                                      uncertain=("uncertain", "sum"), in_evidence=("in_evidence", "sum"))
    q = pd.DataFrame([quality(sid, g) for sid, g in seg.groupby("source_id")]).set_index("source_id")
    meta["text method"] = q["method"].replace({"resmi metin": "official published text"})
    meta["speaker"] = meta["speaker"].str.replace("Fed Başkanı", "Fed Chair").str.replace("TCMB Başkanı", "TCMB Governor")
    meta["match vs official text"] = q["match_rate"]
    meta["published"] = pd.to_datetime(meta["published"]).dt.strftime("%Y-%m-%d %H:%M")
    c = pd.read_csv("reports/speech_claims.csv")
    st = c.groupby(["source_id", "status"]).size().unstack(fill_value=0)
    st.columns = [STATUS_EN.get(x, x) for x in st.columns]
    ver = f[f["in_evidence"]][["source_id", "variable", "values", "tool_value", "status"]].copy()
    ver["values"] = ver["values"].map(lambda v: ", ".join(f"{float(x):g}" for x in v))
    ver["tool_value"] = ver["tool_value"].map(lambda v: ", ".join(f"{float(x):.2f}" for x in str(v).strip("()").split(",")))
    ver = ver.rename(columns={"values": "value in speech", "tool_value": "value from data tool"})
    ver["text (start)"] = f.loc[ver.index, "text"].str.slice(0, 90) + "…"
    return meta, st, ver


# ---------- grafikler ----------
def fig_macro(m: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    style(ax, "y")
    ax.plot(m.index, m["cpi_yoy"], color=C["s2"], linewidth=2, label="CPI, yearly % (as published)")
    ax.plot(m.index, m["policy_rate"], color=C["s1"], linewidth=2, label="TCMB policy rate %")
    ax.plot(m.index, m["fed_upper"], color=C["muted"], linewidth=1.5, linestyle="--", label="Fed upper bound %")
    for i, (d, lab) in enumerate(MARKS):
        d = pd.Timestamp(d)
        ax.axvline(d, color=C["grid"], linewidth=1)
        ax.text(d, 88 if i % 2 == 0 else 77, lab, fontsize=7, color=C["text2"], ha="center", va="top",
                bbox=dict(facecolor=C["surface"], edgecolor="none", pad=1))
    ax.set_ylim(0, 90)
    ax.set_ylabel("%", fontsize=9, color=C["text2"])
    ax.set_title("Figure 1. Negative real rates until mid-2023, then tight policy (point-in-time values)",
                 loc="left", fontsize=10.5, color=C["text"])
    ax.legend(frameon=False, fontsize=8.5, labelcolor=C["text2"], loc="lower right", bbox_to_anchor=(1.0, 0.08))
    return save_fig(fig, "05_macro_timeline.png")


def fig_try_usd(m: pd.DataFrame) -> str:
    xu = get_index_history("XU100")["Close"]
    usd = m["usdtry"].reindex(xu.index).ffill()
    a, b = xu / xu.iloc[0], (xu / usd) / (xu.iloc[0] / usd.iloc[0])
    fig, ax = plt.subplots(figsize=(8, 3.8))
    style(ax, "y")
    ax.plot(a.index, a, color=C["s1"], linewidth=2, label="XU100 in TRY")
    ax.plot(b.index, b, color=C["s2"], linewidth=2, label="XU100 in USD")
    for s, col in ((a, C["s1"]), (b, C["s2"])):
        ax.text(s.index[-1], s.iloc[-1], f"  {s.iloc[-1]:.1f}×", fontsize=8, color=C["text2"], va="center")
    ax.axhline(1, color=C["grid"], linewidth=1)
    ax.set_yscale("log")
    ax.set_ylabel("growth of 1 unit (log scale)", fontsize=9, color=C["text2"])
    ax.set_title("Figure 2. Most of the nominal TRY gain is currency and inflation", loc="left", fontsize=10.5,
                 color=C["text"])
    ax.legend(frameon=False, fontsize=8.5, labelcolor=C["text2"], loc="upper left")
    return save_fig(fig, "05_xu100_try_usd.png")


# ---------- rapor ----------
def build() -> str:
    notes = load_notes(NOTES_FILE)
    m = macro()
    yr, real_note = yearly_table(m)
    rf, strat, on_share = risk_flag(m)
    beta = fx_betas(m)
    src, cats, scope, news_s, contract_top = news_tables()
    meta, claims, verified = speech_tables()
    f1, f2 = fig_macro(m), fig_try_usd(m)
    pct = {c: ("{:+.1f}" if c.startswith(("XU100", "USD")) else "{:.1f}") for c in yr.columns}

    md = f"""# Report 5 — Macro and Contextual Evidence Integration

_{DISCLAIMER}_

Generated by `python -m src.reports.build_reports --only 5` from committed outputs and the cached price data.

{stamp(SOURCES)}

## 1. Research question

Project spec §15–16 and §25: build the macro, news and speech layers so that every value enters a historical
decision **only from the moment it was public**, and decide what each layer may contribute to the evidence. The
questions are (a) what the macro regime was and how much it explains, (b) whether macro or news context improves
decisions in practice, and (c) how unstructured text (news, speeches) can be used without letting it replace data.

{note(notes, "summary")}

## 2. Macro layer: sources and point-in-time alignment

**Table 1. Macro variables**

{md_table(sources_table(), index=False)}

![Figure 1]({f1})

**Table 2. Year by year: the market in TRY, in USD and after inflation**

{md_table(yr, fmt=pct)}

_{real_note}._

![Figure 2]({f2})

{note(notes, "macro")}

## 3. Does the macro risk flag help?

The flag is on in {on_share * 100:.1f}% of trading days. Strategy D is strategy C (fundamental + sector ranking) with half
position size while the flag is on. `D_makro_oynaklik` adds inverse-volatility weights, `D_makro_faiz` also halves
during rate-hiking periods, and `E_haber` adds the KAP news rules (Step 10C).

**Table 3. XU100 after the flag turns on or off (%)**

{md_table(rf, fmt={"XU100 mean if on %": "{:+.2f}", "XU100 mean if off %": "{:+.2f}", "t": "{:.2f}", "p": "{:.3f}"})}

**Table 4. Strategies with and without the macro and news layers (after 0.15% one-way costs)**

{md_table(strat, fmt={"total return %": "{:.0f}", "XU100 %": "{:.0f}", "equal-weight B&H %": "{:.0f}",
                      "max drawdown %": "{:.0f}"})}

{note(notes, "risk_flag")}

## 4. Which sectors are exposed to the lira?

Daily return of each peer group (equal weight) minus XU100, regressed on the daily % change of USD/TRY
(Newey–West standard errors, 5 lags). A beta of +0.1 means: on a day the lira loses 1%, the group beats XU100 by
0.1 pt on average.

**Table 5. Sector FX sensitivity**

{md_table(beta, fmt={"beta to USD/TRY": "{:+.3f}", "t (HAC)": "{:.2f}", "p": "{:.3f}", "q BH": "{:.3f}"})}

{note(notes, "fx")}

## 5. News layer

**Table 6. News sources in the archive**

{md_table(src)}

**Table 7. Most frequent KAP disclosure categories (keyword rules)**

{md_table(cats.to_frame())}

**Table 8. Scope of tagged items (company / sector / macro / general)**

{md_table(scope)}

### 5.1 Do KAP news categories move prices?

Rules fixed **before** looking at outcomes (2026-11-18): negative = management departure, delayed financial
statements, lawsuit/penalty, liquidation/bankruptcy; positive = new contract or order, start of a share buy-back.
Event day = first trading day after publication. Abnormal returns are vs XU100, compared with an any-day baseline.

**Table 9. Price reaction by KAP news type (%)**

{md_table(news_s, fmt={"p (event day)": "{:.4f}", "q (event day)": "{:.3f}"})}

**Table 10. Who files the "new contract" disclosures?**

{md_table(contract_top.to_frame())}

{note(notes, "news")}

## 6. Speech layer

**Table 11. Speech sources**

{md_table(meta.T.rename_axis("field"), fmt={"match vs official text": "{:.3f}"})}

**Table 12. Numerical claims by verification status**

{md_table(claims)}

**Table 13. Claims that entered the committed evidence (verified against deterministic tools)**

{md_table(verified, index=False)}

{note(notes, "speech")}

## 7. How the layers enter the harness

{note(notes, "harness")}

## 8. Türkiye context

{note(notes, "turkey")}

## 9. Limitations

{note(notes, "limitations")}

## 10. Conclusion

{note(notes, "conclusion")}
"""
    return str(write("05_macro_context.md", md))
