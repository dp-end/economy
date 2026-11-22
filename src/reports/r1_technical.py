"""Rapor 1 — Technical reversal-event report (PDF Bölüm 8 ve 13).

Kaynak: reports/technical_events.parquet (Adım 4, her olay tek satır) + reports/event_summary_*.csv.
Bu dosyada yeni olay tanımı yoktur; yalnızca hazır olaylar üzerinde ek kırılımlar hesaplanır:
XU100 yönü ve oynaklık rejimi, sıçrama/başarısızlık sayıları, hacim onayı, ilk yarı / ikinci yarı,
gün kümelemeli t testi ve çoklu test düzeltmesi.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from src.data.prices import get_index_history
from src.events.technical_events import DEDUPE_GAP, HORIZONS, K2_APPLICABLE, KAMA_FLAT, RSI_HIGH, RSI_LOW, baseline
from src.reports.common import (C, DISCLAIMER, bh, clustered_t, load_notes, md_table, note, save_fig, stamp,
                                style, t_test, write)

SOURCES = ["reports/technical_events.parquet", "reports/event_summary_fwd.csv", "reports/event_summary_exc.csv",
           "reports/event_summary_regime.csv"]
NOTES_FILE = "01_technical_reversal.md"

# olay -> (aile, tanım). Tanımlar src/events/technical_events.py: detect_events ile aynı.
EVENTS = {
    "bb_lower_reentry": ("Bollinger", "close was below the lower band (20, 2σ) yesterday and is back inside today"),
    "bb_upper_reentry": ("Bollinger", "close was above the upper band yesterday and is back inside today"),
    "rsi_oversold_exit": ("RSI", "RSI(14) crosses back above 30 (exit from oversold; local-trough proxy)"),
    "rsi_overbought_exit": ("RSI", "RSI(14) crosses back below 70 (exit from overbought; local-peak proxy)"),
    "supertrend_up": ("Supertrend", "Supertrend(10, 3×ATR) direction flips to up"),
    "supertrend_down": ("Supertrend", "Supertrend direction flips to down"),
    "kama_turn_up": ("KAMA", "KAMA(10, 2, 30) slope turns from ≤0 to >0"),
    "kama_turn_down": ("KAMA", "KAMA slope turns from ≥0 to <0"),
    "cloud_break_up": ("Ichimoku", "price moves from inside/below the cloud (9, 26, 52) to above it"),
    "cloud_break_down": ("Ichimoku", "price moves from inside/above the cloud to below it"),
    "resistance_break": ("Support/Resistance", "close crosses above yesterday's nearest pivot resistance"),
    "support_break": ("Support/Resistance", "close crosses below yesterday's nearest pivot support"),
}
REGIME_COLS = ["stock>SMA200", "stock≤SMA200", "XU100 up", "XU100 down", "high vol", "low vol"]


def load_events() -> pd.DataFrame:
    e = pd.read_parquet("reports/technical_events.parquet")
    e["date"] = pd.to_datetime(e["date"])
    xu = get_index_history("XU100")["Close"]
    r = xu.pct_change()
    vol = r.rolling(20).std()
    med = vol.expanding(250).median()
    mk = pd.DataFrame(index=xu.index)
    sma = xu.rolling(200).mean()
    mk["xu_regime"] = np.where(sma.isna(), "unknown", np.where(xu > sma, "XU100 up", "XU100 down"))
    mk["vol_regime"] = np.where(med.isna(), "unknown", np.where(vol > med, "high vol", "low vol"))
    e = e.join(mk, on="date")
    for h in HORIZONS:                                        # yön düzeltmeli, yüzde
        e[f"raw_{h}"] = e["direction"] * e[f"fwd_{h}"] * 100
        e[f"x_{h}"] = e["direction"] * e[f"exc_{h}"] * 100
    return e


def main_table(e: pd.DataFrame, base: pd.Series) -> pd.DataFrame:
    rows = []
    for ev, g in e.groupby("event"):
        t, p = t_test(g["x_5"])
        tc, pc, nd = clustered_t(g["x_5"], g["date"])
        rows.append({"event": ev, "family": EVENTS[ev][0], "dir": "bull" if g["direction"].iloc[0] > 0 else "bear",
                     "n": len(g), "raw 5d %": g["raw_5"].mean(), "raw hit %": (g["raw_5"] > 0).mean() * 100,
                     "excess 5d %": g["x_5"].mean(), "excess hit %": (g["x_5"] > 0).mean() * 100,
                     "t": t, "t (by day)": tc, "p (by day)": pc, "days": nd})
    m = pd.DataFrame(rows).set_index("event")
    m["q BH"] = bh(m["p (by day)"])
    # baz: olaysız herhangi bir gün; boğa olayı için +baz, ayı olayı için -baz "hiçbir şey bilmeden" beklenen
    m["raw vs any-day"] = m["raw 5d %"] - np.where(m["dir"] == "bull", base["mean_5"], -base["mean_5"])
    return m.sort_values(["family", "dir"], ascending=[True, False])


def bounce_table(e: pd.DataFrame) -> pd.DataFrame:
    g = e.groupby("event")
    out = pd.DataFrame({
        "n": g.size(),
        "bounce (5d)": g["raw_5"].apply(lambda s: int((s > 0).sum())),
        "failure (5d)": g["raw_5"].apply(lambda s: int((s <= 0).sum())),
        "median strength 5d %": g["raw_5"].median(),
        "median strength 10d %": g["raw_10"].median(),
        "median excess 5d %": g["x_5"].median(),
        "excess failure %": g["x_5"].apply(lambda s: (s <= 0).mean() * 100),
    })
    return out


def horizon_table(e: pd.DataFrame) -> pd.DataFrame:
    g = e.groupby("event")
    return pd.DataFrame({f"excess {h}d %": g[f"x_{h}"].mean() for h in HORIZONS})


def regime_table(e: pd.DataFrame) -> tuple:
    own = e["regime"].map({"uptrend": "stock>SMA200", "downtrend": "stock≤SMA200"})
    mean, n = {}, {}
    for col in (own, e["xu_regime"], e["vol_regime"]):
        for (ev, lab), g in e.groupby([e["event"], col]):
            if lab in REGIME_COLS:
                mean[(ev, lab)] = g["x_5"].mean()
                n[(ev, lab)] = len(g)
    mean = pd.Series(mean).unstack()[REGIME_COLS].rename_axis("event")
    n = pd.Series(n).unstack()[REGIME_COLS].rename_axis("event")
    return mean, n


def volume_table(e: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for ev, g in e.groupby("event"):
        a, b = g[~g["vol_confirm"]], g[g["vol_confirm"]]
        rows.append({"event": ev, "n (normal vol)": len(a), "excess 5d % (normal)": a["x_5"].mean(),
                     "n (volume z>1)": len(b), "excess 5d % (volume z>1)": b["x_5"].mean(),
                     "difference": b["x_5"].mean() - a["x_5"].mean()})
    return pd.DataFrame(rows).set_index("event")


def diff_test(x: pd.Series, flag: pd.Series, dates: pd.Series) -> float:
    """Koşullu ve koşulsuz olayların ortalama farkı için p: x ~ 1 + flag, gün kümelemeli standart hata."""
    import statsmodels.api as sm
    d = pd.DataFrame({"x": x.to_numpy(), "f": flag.astype(float).to_numpy(), "g": pd.factorize(dates)[0]}).dropna()
    if d["f"].sum() < 2 or (1 - d["f"]).sum() < 2:
        return np.nan
    r = sm.OLS(d["x"], sm.add_constant(d["f"])).fit(cov_type="cluster", cov_kwds={"groups": d["g"]})
    return float(r.pvalues["f"])


def state_table(e: pd.DataFrame) -> pd.DataFrame:
    """Olay anındaki gösterge durumları: olayların yüzde kaçında her onay / ters trend var (betimleyici)."""
    g = e.groupby("event")
    return pd.DataFrame({
        "n": g.size(),
        "RSI zone %": g["conf_rsi"].mean() * 100, "volume z>1 %": g["conf_vol"].mean() * 100,
        "KAMA turning %": g["conf_kama"].mean() * 100,
        "Supertrend against %": g.apply(lambda d: ((d["st_state"] == "asagi") if d["direction"].iloc[0] > 0
                                                   else (d["st_state"] == "yukari")).mean() * 100),
        "cloud against %": g.apply(lambda d: ((d["cloud_pos"] == "altinda") if d["direction"].iloc[0] > 0
                                              else (d["cloud_pos"] == "ustunde")).mean() * 100),
        "all three (n)": g["k1_all3"].sum().astype(int),
        "all three excess 5d %": g.apply(lambda d: d.loc[d["k1_all3"], "x_5"].mean()),
    })


def combined_table(e: pd.DataFrame) -> pd.DataFrame:
    """Önceden yazılmış K1 (≥2 onay) ve K2 (trende karşı): koşullu − koşulsuz 5 günlük fazla getiri."""
    rows = []
    for cond, col, sign in (("K1 confirmed", "k1_confirmed", "+"), ("K2 against trend", "k2_against_trend", "−")):
        for ev, g in e.groupby("event"):
            if col == "k2_against_trend" and ev not in K2_APPLICABLE:
                continue
            on, off = g[g[col]], g[~g[col]]
            rows.append({"condition": cond, "event": ev, "expected": sign, "n with": len(on), "n without": len(off),
                         "excess 5d % with": on["x_5"].mean(), "excess 5d % without": off["x_5"].mean(),
                         "difference": on["x_5"].mean() - off["x_5"].mean(),
                         "p (by day)": diff_test(g["x_5"], g[col], g["date"])})
    return pd.DataFrame(rows)


def test_family(m: pd.DataFrame, e: pd.DataFrame) -> tuple:
    """Raporun tek BH ailesi: 12 olay türü (Tablo 2) + önceden yazılmış K1/K2 (Tablo 6b). Final rapor da bunu kullanır.
    Döner: ('q (all tests)' eklenmiş ana tablo, birleşik koşul tablosu)."""
    ct = combined_table(e)
    q = bh(np.concatenate([m["p (by day)"].to_numpy(), ct["p (by day)"].to_numpy()]))
    m = m.drop(columns="q BH").assign(**{"q (all tests)": q[:len(m)]})
    ct["q (all tests)"] = q[len(m):]
    return m, ct


def oos_table(e: pd.DataFrame) -> tuple:
    split = e["date"].min() + (e["date"].max() - e["date"].min()) / 2
    split = pd.Timestamp(split.date())
    rows = []
    for ev, g in e.groupby("event"):
        a, b = g[g["date"] < split], g[g["date"] >= split]
        rows.append({"event": ev, "n first half": len(a), "excess 5d % first": a["x_5"].mean(),
                     "n second half": len(b), "excess 5d % second": b["x_5"].mean(),
                     "same sign": np.sign(a["x_5"].mean()) == np.sign(b["x_5"].mean())})
    return pd.DataFrame(rows).set_index("event"), split


# ---------- grafikler ----------
def fig_raw_vs_excess(m: pd.DataFrame, base: pd.Series) -> str:
    d = m.sort_values("raw 5d %")
    y = np.arange(len(d))
    fig, ax = plt.subplots(figsize=(8, 5.6))
    style(ax, "x")
    h = 0.38
    ax.barh(y + h / 2, d["raw 5d %"], height=h - 0.04, color=C["s1"], label="raw 5-day return (direction-adjusted)")
    ax.barh(y - h / 2, d["excess 5d %"], height=h - 0.04, color=C["s2"], label="excess vs XU100 (direction-adjusted)")
    ax.axvline(0, color=C["text2"], linewidth=0.8)
    for sgn, lab in ((1, "any day +"), (-1, "any day −")):
        ax.axvline(sgn * base["mean_5"], color=C["muted"], linestyle="--", linewidth=1)
        ax.text(sgn * base["mean_5"], len(d) - 0.3, f"{lab}{base['mean_5']:.2f}%", color=C["text2"], fontsize=8,
                ha="center", va="bottom")
    ax.set_yticks(y, [f"{i} ({'bull' if m.loc[i, 'dir'] == 'bull' else 'bear'})" for i in d.index])
    ax.set_xlabel("mean 5-day return, % (positive = moved in the event's expected direction)", color=C["text2"],
                  fontsize=9)
    ax.set_title("Figure 1. Raw follow-through mostly mirrors market drift; excess return is near zero",
                 loc="left", fontsize=10.5, color=C["text"])
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=C["text2"])
    return save_fig(fig, "01_raw_vs_excess.png")


def fig_regimes(mean: pd.DataFrame, n: pd.DataFrame) -> str:
    cmap = LinearSegmentedColormap.from_list("div", [C["neg"], C["mid"], C["pos"]])
    lim = np.nanmax(np.abs(mean.to_numpy()))
    fig, ax = plt.subplots(figsize=(8, 5.6))
    ax.imshow(mean.to_numpy(), cmap=cmap, vmin=-lim, vmax=lim, aspect="auto")
    for i in range(mean.shape[0]):
        for j in range(mean.shape[1]):
            v, k = mean.iat[i, j], n.iat[i, j]
            txt = "–" if np.isnan(v) else f"{v:+.2f}\n(n={int(k)})"
            dark = not np.isnan(v) and abs(v) > 0.6 * lim
            ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white" if dark else C["text"])
    ax.set_xticks(range(mean.shape[1]), mean.columns, fontsize=8.5, color=C["text2"])
    ax.set_yticks(range(mean.shape[0]), mean.index, fontsize=8.5, color=C["text2"])
    ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("Figure 2. Mean 5-day excess return vs XU100 by regime (%, blue = worked, red = failed)",
                 loc="left", fontsize=10.5, color=C["text"])
    return save_fig(fig, "01_regimes.png")


# ---------- rapor ----------
def build() -> str:
    notes = load_notes(NOTES_FILE)
    raw_all = load_events()
    quake = int(raw_all["quake_window"].sum())
    e = raw_all[~raw_all["quake_window"]].copy()
    base = baseline().iloc[0]
    m = main_table(e, base)
    st_tab = state_table(e).loc[m.index]
    m, ct = test_family(m, e)
    bt = bounce_table(e).loc[m.index]
    ht = horizon_table(e).loc[m.index]
    rm, rn = regime_table(e)
    rm, rn = rm.loc[m.index], rn.loc[m.index]
    vt = volume_table(e).loc[m.index]
    ot, split = oos_table(e)
    ot = ot.loc[m.index]
    f1, f2 = fig_raw_vs_excess(m, base), fig_regimes(rm, rn)

    n_tests = len(m)
    n_family = len(m) + len(ct)
    sig = m[(m["q (all tests)"] < 0.05)]
    ct_sig = ct[ct["q (all tests)"] < 0.05]
    defs = pd.DataFrame([(k, v[0], "bull" if k in m.index and m.loc[k, "dir"] == "bull" else "bear", v[1])
                         for k, v in EVENTS.items()], columns=["event", "family", "dir", "definition"])
    regime_txt = rm.copy()
    for c in regime_txt.columns:
        regime_txt[c] = [f"{v:+.2f} ({int(k)})" if not np.isnan(v) else "–" for v, k in zip(rm[c], rn[c])]

    md = f"""# Report 1 — Technical Reversal Events on 30 BIST 100 Stocks

_{DISCLAIMER}_

Generated by `python -m src.reports.build_reports --only 1` from committed outputs (no analysis is re-run).

{stamp(SOURCES)}

## 1. Research question

Project spec §8 and §13: treat each technical signal as a **timestamped event**, record the market state at that
time and measure what happened over the next 1, 3, 5 and 10 trading days. The question is not "does the indicator
work" but: **after this event, did the stock move in the expected direction more often or by more than it would
have anyway, and does the answer survive regimes, a later sample and multiple-testing correction?**

{note(notes, "summary")}

## 2. Data and method

- Universe: the fixed 30-stock study universe (`config/universe.csv`), daily İş Yatırım prices,
  {e['date'].min().date()} – {e['date'].max().date()}.
- Events are detected using only today's and earlier data (`src/events/technical_events.py: detect_events`).
  If the same event repeats within {DEDUPE_GAP} trading days for a stock, only the first is kept.
- **Direction-adjusted return**: for a bullish event the forward return, for a bearish event minus the forward
  return. Positive always means "moved the way the event suggested".
- **Raw** = stock's own forward return. **Excess** = stock return minus XU100 return over the same horizon.
- **Any-day baseline**: forward return from *every* stock-day (no event), the "knowing nothing" benchmark.
- The Kahramanmaraş earthquake window (events whose 10-day horizon touches 6–24 Feb 2023, {quake} events) is
  excluded; Borsa İstanbul was closed 8–14 Feb 2023 and prices were not comparable.
- Regimes: the stock above/below its own 200-day SMA; XU100 above/below its 200-day SMA; XU100 20-day volatility
  above/below its expanding median (needs 250 days, earlier rows are "unknown" and left out of those columns).
- Statistics: t-test of the mean; because many stocks fire on the same day, a **day-clustered t** (mean per day
  first) is reported and used for p-values. Benjamini–Hochberg q is computed once over all {n_family} tests of
  this report: the {n_tests} event types (Table 2) and the pre-registered combined conditions (Table 6b).

**Table 1. Event definitions**

{md_table(defs, index=False)}

## 3. Main result: event follow-through at 5 days

Any-day baseline (all stock-days, no event): 5-day mean **{base['mean_5']:+.2f}%**, positive in
**{base['hit_5']:.1f}%** of cases (1d {base['mean_1']:+.2f}%, 10d {base['mean_10']:+.2f}%).

**Table 2. Direction-adjusted 5-day returns (%), earthquake window excluded**

{md_table(m.drop(columns=["family"]), fmt={"p (by day)": "{:.3f}", "q (all tests)": "{:.3f}", "t": "{:.2f}",
                                            "t (by day)": "{:.2f}"})}

"raw vs any-day" = raw mean minus what the baseline drift alone gives in that direction
(+{base['mean_5']:.2f}% for bullish, −{base['mean_5']:.2f}% for bearish events).
Events with q < 0.05 on excess return (day-clustered): **{', '.join(sig.index) if len(sig) else 'none'}**.

![Figure 1]({f1})

{note(notes, "main")}

## 4. Bounce / failure counts and reversal strength

A **bounce** is a positive direction-adjusted raw 5-day return, a **failure** is zero or negative.
Reversal strength is the median direction-adjusted return (medians resist a few extreme days).

**Table 3. Bounce and failure counts**

{md_table(bt, fmt={"median strength 5d %": "{:.2f}", "median strength 10d %": "{:.2f}",
                   "median excess 5d %": "{:.2f}", "excess failure %": "{:.1f}"})}

{note(notes, "bounce")}

## 5. Horizons

**Table 4. Mean direction-adjusted excess return vs XU100 by horizon (%)**

{md_table(ht)}

{note(notes, "horizons")}

## 6. Regimes

**Table 5. Mean 5-day excess return (%) and event count by regime**

{md_table(regime_txt)}

![Figure 2]({f2})

{note(notes, "regimes")}

## 7. Combined condition: volume confirmation

The spec's ASELS example asks whether events with confirming conditions behave differently than single-indicator
events. This section uses one confirmation flag, 20-day volume z-score > 1 on the event day (not tested);
Section 7b adds the full indicator state and the pre-registered tests.

**Table 6. Excess 5-day return with and without volume confirmation (%)**

{md_table(vt)}

{note(notes, "volume")}

## 7b. Indicator states at the event and pre-registered combined conditions

The spec's event record (§8) stores the market state at the event: RSI state, Supertrend state, KAMA slope,
Ichimoku position and volume. Every event now carries these states, measured at the event-day close with data up
to that day only (`src/events/technical_events.py: states`).

**Written on 22 November 2026, before any return of these conditions was computed** (only the counts of events
were looked at, to make sure each condition occurs often enough to be tested). For a bullish event, with the mirror
image for a bearish event:

- **Confirmations.** (1) RSI(14) was ≤ {RSI_LOW} (bearish: ≥ {RSI_HIGH}) on at least one of the last 3 days;
  (2) volume z-score > 1 on at least one of the last 3 days; (3) **KAMA turning**: the 5-day KAMA change is higher
  than it was 5 days earlier (bearish: lower), i.e. the decline is flattening.
- **K1 "confirmed"**: at least two of the three confirmations. Expected: stronger follow-through than the same
  event without confirmation (difference > 0).
- **K2 "against the trend"**: Supertrend points against the event and the price is on the opposite side of the
  Ichimoku cloud (bullish event below a bearish Supertrend and below the cloud). Expected: weaker follow-through
  (difference < 0). It cannot occur for Supertrend and cloud events themselves, so it is tested on
  {len(K2_APPLICABLE)} event types.
- **Why not the spec's literal triple?** The spec's example (RSI trough + rising volume + flattening KAMA, all at
  once) occurs only a handful of times per event type (Table 6a, "all three"). With RSI judged on the event day
  itself and KAMA required to be flat (|5-day change| ≤ {KAMA_FLAT}%), it occurred in **none** of the
  {len(e):,} events: when RSI is at a trough the price has just fallen, so KAMA is still falling. It is reported, not
  tested.
- **Test.** Difference in mean direction-adjusted 5-day excess return (vs XU100) between events with and without
  the condition, standard errors clustered by day; two-sided p; one Benjamini–Hochberg family with Table 2.

**Table 6a. Indicator states at the event (% of events; descriptive)**

{md_table(st_tab, fmt={"n": "{:,.0f}", "RSI zone %": "{:.0f}", "volume z>1 %": "{:.0f}", "KAMA turning %": "{:.0f}",
                       "Supertrend against %": "{:.0f}", "cloud against %": "{:.0f}", "all three (n)": "{:,.0f}",
                       "all three excess 5d %": "{:+.2f}"})}

**Table 6b. Pre-registered combined conditions: 5-day excess return with vs without (%)**

{md_table(ct.set_index(["condition", "event"]), fmt={"n with": "{:,.0f}", "n without": "{:,.0f}",
          "excess 5d % with": "{:+.2f}", "excess 5d % without": "{:+.2f}", "difference": "{:+.2f}",
          "p (by day)": "{:.3f}", "q (all tests)": "{:.3f}"})}

Conditions with q < 0.05: **{', '.join(f"{r.condition} / {r.event}" for r in ct_sig.itertuples()) or 'none'}**.

{note(notes, "combined")}

## 8. Out-of-sample check: first half vs second half

The sample is split at **{split.date()}**. A pattern worth trusting should keep at least its sign.

**Table 7. Mean 5-day excess return (%) in each half**

{md_table(ot)}

Same sign in both halves: **{int(ot['same sign'].sum())} of {len(ot)}** event types.

{note(notes, "oos")}

## 9. Multiple testing

This report alone looks at {n_tests} event types × {len(HORIZONS)} horizons × 2 return definitions
(raw, excess) plus {len(REGIME_COLS)} regime cuts, more than 100 numbers. At a 5% level, about 1 in 20 would look
"significant" by chance. That is why Table 2 uses day-clustered p-values and Benjamini–Hochberg q, and why a
result is only called a finding when it also keeps its sign out of sample (Table 7). The {n_family} formal tests
(Tables 2 and 6b) share one Benjamini–Hochberg correction.

{note(notes, "multiple")}

## 10. Türkiye context

{note(notes, "turkey")}

## 11. How the harness uses these results

{note(notes, "harness")}

## 12. Limitations

{note(notes, "limitations")}

## 13. Conclusion

{note(notes, "conclusion")}
"""
    return str(write("01_technical_reversal.md", md))
