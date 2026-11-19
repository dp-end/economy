"""Rapor 4 — Quarterly fundamentals and post-disclosure price reaction (PDF Bölüm 9 ve 14).

Kaynak: reports/earnings_events.parquet (Adım 7C, KAP yayın tarihleriyle hizalı) + earnings_*.csv +
kap_disclosure_dates.csv + Fintables tabloları (load_all) + fiyat önbelleği.
Ek kırılımlar: gelir / faaliyet kârı / net marj / ROE değişimi üçte birleri, sektör sepetine göre anormal getiri,
bilanço x teknik durum, kâr yönü kategorileri, finans / finans dışı. Tüm testler tek BH ailesinde.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.data.fundamentals import load_all
from src.data.prices import get_index_history, get_stock_history
from src.data.universe import build_sector_peer_groups
from src.events.earnings_reaction import WINDOWS, add_terciles, baseline
from src.harness.evidence import profit_direction
from src.indicators.features import full_features
from src.reports.common import C, DISCLAIMER, bh, load_notes, md_table, note, save_fig, stamp, style, write

SOURCES = ["reports/earnings_events.parquet", "reports/earnings_by_tercile.csv", "reports/earnings_spread_tests.csv",
           "reports/earnings_spread_by_quarter.csv", "reports/kap_disclosure_dates.csv",
           "reports/disclosure_spike_check.csv"]
NOTES_FILE = "04_earnings_reaction.md"
SHOW = ["pre5", "d0", "d0_4", "post1_20"]
LABEL = {"pre5": "5 days before", "d0": "event day", "d0_1": "days 0–1", "d0_4": "days 0–4", "post1_5": "days 1–5",
         "post1_20": "days 1–20"}
TERC = {"alt": "bottom", "orta": "middle", "ust": "top"}
TMS29 = pd.Timestamp("2023-12-31")      # enflasyon muhasebesi (TMS 29) ilk kez 2023 yıl sonu tablolarında
PATH = range(-5, 21)


def _yoy(a: pd.DataFrame, col: str) -> pd.Series:
    """Aynı tarihli 4 çeyrek önceki değere göre değişim; eksik çeyrek varsa NaN (kaydırma değil tarih eşleşmesi)."""
    out = []
    for t, g in a.groupby("ticker"):
        s = g.set_index("period_end")[col]
        prev = s.copy()
        prev.index = prev.index + pd.offsets.QuarterEnd(4)
        prev = prev.reindex(s.index)
        out.append(pd.Series((s - prev).to_numpy() / (prev.abs().to_numpy() if col not in ("net_margin", "roe_q")
                                                       else 1.0), index=g.index))
    return pd.concat(out).replace([np.inf, -np.inf], np.nan)


def load_events() -> pd.DataFrame:
    e = pd.read_parquet("reports/earnings_events.parquet")
    e = e[e["event_day"].notna() & ~e["quake_window"]].copy()
    for c in ("event_day", "period_end"):
        e[c] = pd.to_datetime(e[c])
    a = pd.concat(load_all(), names=["ticker"]).reset_index()
    a["period_end"] = pd.to_datetime(a["period_end"])
    a = a.sort_values(["ticker", "period_end"]).reset_index(drop=True)
    a["revenue_yoy"] = _yoy(a, "revenue")
    a["op_income_yoy"] = _yoy(a, "op_income")
    a["margin_chg"] = _yoy(a, "net_margin")            # puan farkı (oran değil)
    a["roe_chg"] = _yoy(a, "roe_q")
    prev = a[["ticker", "period_end", "net_income"]].copy()
    prev["period_end"] = prev["period_end"] + pd.offsets.QuarterEnd(4)
    a = a.merge(prev.rename(columns={"net_income": "ni_prev"}), on=["ticker", "period_end"], how="left")
    e = e.merge(a[["ticker", "period_end", "revenue_yoy", "op_income_yoy", "margin_chg", "roe_chg", "ni_prev"]],
                on=["ticker", "period_end"], how="left")
    e["direction"] = [profit_direction(c, p) if pd.notna(c) and pd.notna(p) else None
                      for c, p in zip(e["net_income"], e["ni_prev"])]
    # teknik durum: açıklamadan önceki son kapanış (event_day - 1) 200 günlük ortalamanın üstünde mi
    above = {}
    for t in e["ticker"].unique():
        f = full_features(t)
        above[t] = (f["Close"] > f["sma_200"]).where(f["sma_200"].notna()).shift(1)
    e["above_sma200"] = [above[t].get(d, np.nan) for t, d in zip(e["ticker"], e["event_day"])]
    e = sector_abnormal(e)
    for k in WINDOWS:
        e[k] = e[k] * 100
    return e


def sector_abnormal(e: pd.DataFrame) -> pd.DataFrame:
    """Aynı pencerede hisse getirisi eksi akran sepeti (kendisi hariç, eşit ağırlık) getirisi."""
    groups, _ = build_sector_peer_groups()
    group_of = {t: g for g, m in groups.items() for t in m}
    idx = get_index_history("XU100").index
    panel = pd.DataFrame({t: get_stock_history(t)["Close"] for m in groups.values() for t in m}).reindex(idx)
    vals = {k: [] for k in SHOW}
    for t, d in zip(e["ticker"], e["event_day"]):
        peers = [p for p in groups.get(group_of.get(t), []) if p != t]
        pos = idx.get_loc(d) if d in idx else None
        for k in SHOW:
            a, b = WINDOWS[k]
            if pos is None or not peers or pos + a - 1 < 0 or pos + b >= len(idx):
                vals[k].append(np.nan)
                continue
            lo, hi = idx[pos + a - 1], idx[pos + b]
            r = panel.loc[hi] / panel.loc[lo] - 1
            vals[k].append((r[t] - r[peers].mean()) * 100)
    for k in SHOW:
        e[f"{k}_vs_peers"] = vals[k]
    return e


def quarter_spread(e: pd.DataFrame, col: str, windows=SHOW) -> pd.DataFrame:
    d = add_terciles(e, col)
    g = d.dropna(subset=["tercile"]).groupby(["period_end", "tercile"])[list(windows)].mean()
    return (g.xs("ust", level="tercile") - g.xs("alt", level="tercile")).dropna(how="all")


def t_over_quarters(s: pd.Series) -> tuple:
    s = s.dropna()
    if len(s) < 3:
        return np.nan, np.nan, len(s)
    t = s.mean() / (s.std(ddof=1) / math.sqrt(len(s)))
    return t, 2 * stats.t.sf(abs(t), len(s) - 1), len(s)


def metric_table(e: pd.DataFrame) -> pd.DataFrame:
    nonfin = e[~e["financial"]]
    metrics = [("revenue YoY (non-financial)", "revenue_yoy", nonfin),
               ("operating income YoY (non-financial)", "op_income_yoy", nonfin),
               ("net margin change, pt (non-financial)", "margin_chg", nonfin),
               ("ROE change, pt", "roe_chg", e)]
    rows = []
    for name, col, sub in metrics:
        sp = quarter_spread(sub, col)
        half = len(sp) // 2
        for k in ("d0", "d0_4", "post1_20"):
            t, p, nq = t_over_quarters(sp[k])
            rows.append({"metric": name, "window": LABEL[k], "events": int(sub[col].notna().sum()), "quarters": nq,
                         "top − bottom (pt)": sp[k].mean(), "first half": sp[k].iloc[:half].mean(),
                         "second half": sp[k].iloc[half:].mean(), "t": t, "p": p})
    return pd.DataFrame(rows)


def car_path(e: pd.DataFrame) -> pd.DataFrame:
    """Üçte birlere göre ortalama kümülatif anormal getiri (XU100'e göre), açıklamadan 5 gün önce -> 20 gün sonra."""
    xu = get_index_history("XU100")["Close"]
    idx = xu.index
    paths = []
    for t, g in e.dropna(subset=["tercile"]).groupby("ticker"):
        c = get_stock_history(t)["Close"].reindex(idx)
        for d, terc in zip(g["event_day"], g["tercile"]):
            pos = idx.get_loc(d)
            base = pos - 6
            if base < 0 or pos + max(PATH) >= len(idx):
                continue
            row = {k: ((c.iloc[pos + k] / c.iloc[base] - 1) - (xu.iloc[pos + k] / xu.iloc[base] - 1)) * 100
                   for k in PATH}
            row["tercile"] = terc
            paths.append(row)
    p = pd.DataFrame(paths)
    return p.groupby("tercile")[list(PATH)].mean().T


# ---------- grafikler ----------
def fig_car(path: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    style(ax, "y")
    for terc, col in (("ust", C["s1"]), ("orta", C["muted"]), ("alt", C["s2"])):
        ax.plot(path.index, path[terc], color=col, linewidth=2, label=f"{TERC[terc]} third (net income YoY)")
        ax.text(path.index[-1] + 0.3, path[terc].iloc[-1], f"{path[terc].iloc[-1]:+.1f}", va="center", fontsize=8,
                color=C["text2"])
    ax.axvline(-0.5, color=C["text2"], linewidth=0.8, linestyle="--")
    ax.text(-0.4, ax.get_ylim()[1], " first trading day after KAP disclosure", fontsize=8, color=C["text2"], va="top")
    ax.axhline(0, color=C["grid"], linewidth=1)
    ax.set_xlabel("trading days relative to the event day", fontsize=9, color=C["text2"])
    ax.set_ylabel("cumulative return minus XU100, %", fontsize=9, color=C["text2"])
    ax.set_title("Figure 1. Prices separate before and on the event day; the later drift is not reliable (Table 2)", loc="left",
                 fontsize=10.5, color=C["text"])
    ax.legend(frameon=False, fontsize=8.5, labelcolor=C["text2"], loc="upper left", bbox_to_anchor=(0.0, 0.93))
    return save_fig(fig, "04_car_path.png")


def fig_quarters(sp: pd.DataFrame) -> str:
    s = sp["d0_4"]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    style(ax, "y")
    x = np.arange(len(s))
    colors = [C["s2"] if d >= TMS29 else C["s1"] for d in s.index]
    ax.bar(x, s.to_numpy(), color=colors, width=0.7)
    ax.axhline(0, color=C["text2"], linewidth=0.8)
    ax.axhline(s.mean(), color=C["muted"], linewidth=1, linestyle="--")
    ax.text(len(s) - 0.5, s.mean(), f" mean {s.mean():+.2f}", fontsize=8, color=C["text2"], va="bottom", ha="right")
    ax.set_xticks(x, [f"{d.year}Q{d.quarter}" for d in s.index], rotation=60, fontsize=8)
    ax.set_ylabel("top − bottom third, days 0–4, pt", fontsize=9, color=C["text2"])
    ax.set_title("Figure 2. Reaction spread by quarter (orange = inflation-accounting statements, TMS 29)",
                 loc="left", fontsize=10.5, color=C["text"])
    return save_fig(fig, "04_quarter_spread.png")


# ---------- rapor ----------
def build() -> str:
    notes = load_notes(NOTES_FILE)
    raw = pd.read_parquet("reports/earnings_events.parquet")
    e = load_events()
    base = baseline()
    kap = pd.read_csv("reports/kap_disclosure_dates.csv", parse_dates=["publish_time", "period_end"])

    def by(col, sub=None, order=None, labels=None):
        d = (sub if sub is not None else e).dropna(subset=[col])
        g = d.groupby(col)
        out = pd.DataFrame({"n": g.size(), **{f"{LABEL[k]} %": g[k].mean() for k in SHOW},
                            "event-day hit %": g["d0"].apply(lambda s: (s.dropna() > 0).mean() * 100)})
        if order:
            out = out.reindex([o for o in order if o in out.index])
        if labels:
            out.index = [labels.get(i, i) for i in out.index]
        return out.rename_axis(col)

    terc = by("tercile", order=["alt", "orta", "ust"], labels=TERC)
    base_row = pd.DataFrame({"n": [np.nan], **{f"{LABEL[k]} %": [base[k]] for k in SHOW},
                             "event-day hit %": [np.nan]}, index=["any stock-day (baseline)"])
    terc = pd.concat([terc, base_row]).rename_axis("tercile")
    peers = e.dropna(subset=["tercile"]).groupby("tercile")[[f"{k}_vs_peers" for k in SHOW]].mean()
    peers = peers.reindex(["alt", "orta", "ust"]).rename(index=TERC,
                                                       columns={f"{k}_vs_peers": f"{LABEL[k]} %" for k in SHOW})
    sp_tests = pd.read_csv("reports/earnings_spread_tests.csv").set_index("pencere")
    sp_tests.index = pd.Index([LABEL[i] for i in sp_tests.index], name="window")
    sp_tests = sp_tests.rename(columns={"n_ceyrek": "quarters", "ort_fark": "top − bottom (pt)", "pozitif_ceyrek":
                                        "positive quarters", "ilk_yari": "first half", "ikinci_yari": "second half",
                                        "q_bh": "q (6 windows)", "p_bonf": "p Bonferroni"})
    sp_q = pd.read_csv("reports/earnings_spread_by_quarter.csv", parse_dates=["period_end"]).set_index("period_end")
    peer_sp = quarter_spread(e.rename(columns={f"{k}_vs_peers": f"{k}p" for k in SHOW}), "ni_yoy",
                             windows=[f"{k}p" for k in SHOW])
    met = metric_table(e)
    fin = by("financial", labels={True: "financial (AKBNK, ISMEN, TURSG)", False: "non-financial"})
    loss = by("loss", labels={True: "reported a loss", False: "reported a profit"})
    dirn = by("direction", order=["kâr arttı", "kâr azaldı", "zarardan kâra geçti", "kârdan zarara geçti",
                                   "zarar küçüldü (hâlâ zarar)", "zarar büyüdü"],
              labels={"kâr arttı": "profit up", "kâr azaldı": "profit down", "zarardan kâra geçti": "loss → profit",
                      "kârdan zarara geçti": "profit → loss", "zarar küçüldü (hâlâ zarar)": "loss narrowed",
                      "zarar büyüdü": "loss widened"})
    tech = (e.dropna(subset=["tercile", "above_sma200"])
            .assign(state=lambda x: np.where(x["above_sma200"].astype(bool), "above SMA200", "below SMA200"))
            .groupby(["tercile", "state"])
            .agg(n=("d0", "size"), d0=("d0", "mean"), d0_4=("d0_4", "mean"), post=("post1_20", "mean"))
            .rename(columns={"d0": "event day %", "d0_4": "days 0–4 %", "post": "days 1–20 %"}))
    tech.index = tech.index.set_levels([TERC[x] for x in tech.index.levels[0]], level=0)
    tech = tech.reindex(["bottom", "middle", "top"], level=0)
    spike = pd.read_csv("reports/disclosure_spike_check.csv").set_index("olcu")
    spike.columns = ["in reporting season", "outside season", "ratio"]
    spike.index = pd.Index(["volume z-score > 2", "volume z-score > 3", "abs. abnormal return > 4%",
                            "abs. abnormal return > 6%", "both (z > 2 and > 4%)"], name="share of stock-days with")

    # tek BH ailesi: 6 pencere (net kâr) + 4 pencere (akranlara göre) + 15 metrik testi
    fam = [("net income spread vs XU100", w, p) for w, p in zip(sp_tests.index, sp_tests["p"])]
    for k in SHOW:
        t, p, _ = t_over_quarters(peer_sp[f"{k}p"])
        fam.append(("net income spread vs peers", LABEL[k], p))
    fam += [("metric spreads", f"{m} / {w}", p) for m, w, p in zip(met["metric"], met["window"], met["p"])]
    fam = pd.DataFrame(fam, columns=["family", "test", "p"])
    fam["q (all tests)"] = bh(fam["p"])
    met["q (all tests)"] = fam.loc[fam["family"] == "metric spreads", "q (all tests)"].to_numpy()
    peer_tab = []
    for k in SHOW:
        t, p, nq = t_over_quarters(peer_sp[f"{k}p"])
        peer_tab.append({"window": LABEL[k], "quarters": nq, "top − bottom vs peers (pt)": peer_sp[f"{k}p"].mean(),
                         "t": t, "p": p})
    peer_tab = pd.DataFrame(peer_tab).set_index("window")
    peer_tab["q (all tests)"] = fam.loc[fam["family"] == "net income spread vs peers", "q (all tests)"].to_numpy()
    survivors = fam[fam["q (all tests)"] < 0.05].sort_values("p")

    path = car_path(e)
    f1, f2 = fig_car(path), fig_quarters(sp_q)
    pf = lambda x: {c: "{:.3f}" for c in x}  # noqa: E731

    md = f"""# Report 4 — Quarterly Fundamentals and Post-Disclosure Price Reaction

_{DISCLAIMER}_

Generated by `python -m src.reports.build_reports --only 4` from committed outputs, Fintables statements and the
cached price data.

{stamp(SOURCES)}

## 1. Research question

Project spec §9 and §14: align quarterly financial statements with the moment they became **public**, then ask
whether specific changes (profit, revenue, operating profit, margins, ROE) were historically associated with
**different price reactions**: before, on and after the disclosure, against XU100 and against the sector's peers.
The question is not "good earnings = price rises".

{note(notes, "summary")}

## 2. Data and point-in-time alignment

- **Statements.** Fintables quarterly income statements and balance sheets for the 30 stocks (Step 2), 2022Q1 –
  2026Q3. Banks and other financials (AKBNK, ISMEN, TURSG) have no revenue/operating-profit lines, so
  industrial ratios are computed only for non-financial companies.
- **Disclosure time.** The real KAP publication timestamp of each statement (Step 7B, {len(kap)} filings). The
  median filing comes {kap['lag_days'].median():.0f} days after the quarter end, and **{(kap['publish_time'].dt.hour >= 18).mean() * 100:.0f}%
  are published after 18:00**, after the market closes. The **event day** is therefore the first trading day *after*
  publication, the first day the market could react.
- **Legal deadlines are not enough.** {kap['after_deadline'].mean() * 100:.0f}% of filings came after the standard
  deadline, mostly 2023Q4 – 2024Q2 (the first inflation-accounting statements, when SPK extended deadlines). Using
  deadlines instead of real dates would have put information into the wrong days.
- **Events.** {len(raw)} disclosures. {int(raw['event_day'].isna().sum())} have no event day in the price data,
  {int(raw['quake_window'].sum())} fall in the earthquake window, leaving **{len(e)}**.
- **Ranking.** Each quarter, the 30 stocks are ranked by year-over-year net-income change into thirds. Ranking
  within the same quarter removes the common inflation effect (in 2022 the median company's profit roughly doubled
  in nominal TRY). A quarter needs ≥ 15 ranked stocks. 2022Q4 and 2023Q4 drop out, because Fintables lacks the 2022Q4
  quarter for 24 of 30 stocks.
- **Returns.** Abnormal return = stock return − XU100 return over the window, and separately stock return − the
  equal-weight return of its sector peers. Windows are in trading days, with the event day = 0.
- **Statistics.** Each quarter's top-minus-bottom spread is one observation (17 quarters), because stocks in the
  same quarter are not independent. All {len(fam)} tests in this report share one Benjamini–Hochberg correction.

## 3. Main result: reaction by net-income ranking

**Table 1. Mean abnormal return vs XU100 by net-income YoY third (%)**

{md_table(terc, fmt={"n": "{:,.0f}", "event-day hit %": "{:.1f}"})}

![Figure 1]({f1})

**Table 2. Top-minus-bottom spread, one observation per quarter (pt vs XU100)**

{md_table(sp_tests, fmt={"p": "{:.4f}", "q (6 windows)": "{:.3f}", "p Bonferroni": "{:.3f}", "t": "{:.2f}"})}

{note(notes, "main")}

## 4. Quarter by quarter

![Figure 2]({f2})

{note(notes, "quarters")}

## 5. Against sector peers

**Table 3. Mean abnormal return vs the sector peer basket by net-income third (%)**

{md_table(peers)}

**Table 4. Top-minus-bottom spread vs peers, one observation per quarter**

{md_table(peer_tab, fmt={"p": "{:.4f}", "q (all tests)": "{:.3f}", "t": "{:.2f}"})}

{note(notes, "peers")}

## 6. Which fundamental change does the market react to?

**Table 5. Top-minus-bottom third spread by metric (pt vs XU100, one observation per quarter).** Net income is
in Table 2; it is not repeated here so that it is not counted twice in the multiple-testing correction.

{md_table(met.set_index(["metric", "window"]), fmt={"p": "{:.4f}", "q (all tests)": "{:.3f}", "t": "{:.2f}"})}

{note(notes, "metrics")}

## 7. Profit direction, losses and financials

The "+48%" for a company whose loss shrank (PETKM, Step 14) showed that a percentage alone can mislead. Here are
the same words the harness now uses (`profit_direction`).

**Table 6. By direction of quarterly net income vs the same quarter a year earlier (%)**

{md_table(dirn, fmt={"event-day hit %": "{:.1f}"})}

**Table 7. Loss vs profit, financial vs non-financial (%)**

{md_table(pd.concat([loss, fin]).rename_axis("group"), fmt={"event-day hit %": "{:.1f}"})}

{note(notes, "direction")}

## 8. Strong fundamentals with weak technicals, and vice versa

The technical state is measured at the close **before** the event day: is the price above its 200-day average?

**Table 8. Net-income third × technical state (abnormal return vs XU100, %)**

{md_table(tech)}

{note(notes, "tech")}

## 9. Is there unusual activity in reporting season?

Step 7 compared how often large volume spikes and large abnormal moves happen inside reporting windows vs
outside them (share of stock-days).

**Table 9. Spikes inside vs outside reporting season**

{md_table(spike, fmt={"in reporting season": "{:.4f}", "outside season": "{:.4f}", "ratio": "{:.2f}"})}

{note(notes, "spikes")}

## 10. Multiple testing

{len(fam)} tests were corrected together; **{len(survivors)}** have q < 0.05:

{md_table(survivors, index=False, fmt={"p": "{:.5f}", "q (all tests)": "{:.3f}"})}

{note(notes, "multiple")}

## 11. Türkiye context

{note(notes, "turkey")}

## 12. How the harness uses these results

{note(notes, "harness")}

## 13. Limitations

{note(notes, "limitations")}

## 14. Conclusion

{note(notes, "conclusion")}
"""
    return str(write("04_earnings_reaction.md", md))
