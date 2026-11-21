"""Rapor 2 — Sector laggard / catch-up research report (PDF Bölüm 11).

Kaynak: reports/sector_events.parquet (Adım 5) + reports/sector_summary_*.csv + reports/backtest_*.csv (Adım 9).
Ek kırılımlar (yeni olay tanımı yok): harness catch-up koşulları (sektör güçlü + toparlanıyor + zayıf bilanço değil),
yapısal/geçici ayrımı (olay gününde görünen son bilanço üçte biri, makro risk bayrağı), yanlış pozitifler,
ilk yarı / ikinci yarı, gün kümelemeli t testi ve BH düzeltmesi.
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest.strategies import macro_scale
from src.data.universe import build_sector_peer_groups
from src.events.sector_catchup import CONFIGS, EVENT_GAP, FWD_H, LIMIT, _context, baseline
from src.indicators.features import full_features
from src.reports.common import (C, DISCLAIMER, bh, clustered_t, load_notes, md_table, note, save_fig, stamp,
                                style, write)

SOURCES = ["reports/sector_events.parquet", "reports/sector_summary_gap.csv", "reports/sector_summary_by_group.csv",
           "reports/earnings_events.parquet", "reports/backtest_strategies.csv", "reports/backtest_cost_sensitivity.csv"]
NOTES_FILE = "02_sector_catchup.md"
MAIN = "lag10_5pct"                  # harness'in kullandığı yapılandırma (evidence.LAG_THR, 10 gün)
SPLIT = pd.Timestamp("2024-05-12")   # Rapor 1 ile aynı bölme tarihi


def load_events() -> pd.DataFrame:
    """Deprem penceresi hariç olaylar + koşul sütunları (hepsi olay günü kapanışında bilinen veri)."""
    c = _context()
    s = pd.read_parquet("reports/sector_events.parquet")
    s["date"] = pd.to_datetime(s["date"])
    s = s[~s["quake_window"]].copy()
    look = {cfg: n for cfg, (n, _) in CONFIGS.items()}
    own = {n: c["panel"].pct_change(n) for n in set(look.values())}
    s["own_ret"] = [own[look[k]].at[d, t] for k, d, t in zip(s["config"], s["date"], s["ticker"])]
    s["sector_median"] = s["own_ret"] - s["lag"]                       # lag = kendi getirisi - akran ortancası
    s["sector_strong"] = s["sector_median"] > 0
    rec = {}
    for t in s["ticker"].unique():
        f = full_features(t)
        rec[t] = (f["Close"] > f["kama"]) | (f["st_dir"] > 0)          # harness: KAMA üstü ya da Supertrend yukarı
    s["recovering"] = [bool(rec[t].get(d, False)) for t, d in zip(s["ticker"], s["date"])]
    ev = pd.read_parquet("reports/earnings_events.parquet")[["ticker", "event_day", "tercile", "loss"]]
    ev = ev.dropna(subset=["event_day"]).assign(event_day=lambda x: pd.to_datetime(x["event_day"]))
    s = pd.merge_asof(s.sort_values("date"), ev.sort_values("event_day"), left_on="date", right_on="event_day",
                      by="ticker", direction="backward")              # yalnızca olay gününe kadar açıklanmış bilanço
    # bilanço yok / bilanço var ama sıralanmadı (TMS 29 geçişi çeyrekleri) ayrı tutulur
    s["tercile"] = np.where(s["event_day"].isna(), "none", s["tercile"].fillna("unranked"))
    s["macro_risk"] = macro_scale(pd.DatetimeIndex(s["date"])).to_numpy() < 1
    s["harness_catch_up"] = (~s["shock"]) & s["sector_strong"] & s["recovering"] & (s["tercile"] != "alt")
    for h in FWD_H:
        for k in ("gap", "exc", "fwd"):
            s[f"{k}_{h}"] = s[f"{k}_{h}"] * 100
    return s


def stat_row(g: pd.DataFrame, h: int = 10, kind: str = "gap") -> dict:
    x = g[f"{kind}_{h}"].dropna()
    t, p, nd = clustered_t(x, g.loc[x.index, "date"])
    return {"n": len(g), f"{kind} 5d": g[f"{kind}_5"].mean(), f"{kind} 10d": g[f"{kind}_10"].mean(),
            f"{kind} 20d": g[f"{kind}_20"].mean(), "hit 10d %": (x > 0).mean() * 100, "t (by day)": t,
            "p (by day)": p, "days": nd}


def main_table(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (cfg, side, shock), g in s.groupby(["config", "side", "shock"]):
        rows.append({"config": cfg, "side": side, "shock": bool(shock), **stat_row(g)})
    m = pd.DataFrame(rows).set_index(["config", "side", "shock"])
    m["q BH"] = bh(m["p (by day)"])
    return m


def filter_ladder(s: pd.DataFrame, base: pd.Series) -> pd.DataFrame:
    g = s[(s["config"] == MAIN) & (s["side"] == "laggard")]
    steps = [("all laggards", g), ("no shock", g[~g["shock"]]),
             ("+ sector strong", g[~g["shock"] & g["sector_strong"]]),
             ("+ recovering (KAMA/Supertrend)", g[~g["shock"] & g["sector_strong"] & g["recovering"]]),
             ("+ not bottom earnings tercile = harness rule", g[g["harness_catch_up"]])]
    rows = [{"step": "any stock-day (baseline)", "n": np.nan, "gap 5d": base["mean_5"], "gap 10d": base["mean_10"],
             "gap 20d": base["mean_20"]}]
    for name, x in steps:
        r = stat_row(x)
        rows.append({"step": name, **{k: r[k] for k in ("n", "gap 5d", "gap 10d", "gap 20d", "hit 10d %",
                                                         "t (by day)", "p (by day)")}})
    return pd.DataFrame(rows).set_index("step")


def split_table(s: pd.DataFrame, col: str, labels: dict) -> pd.DataFrame:
    """Satırlar labels sözlüğünün sırasıyla (sözlükte olmayan değerler sona)."""
    g = s[(s["config"] == MAIN) & (s["side"] == "laggard") & ~s["shock"]]
    rows = []
    for v, x in g.groupby(col):
        r = stat_row(x)
        rows.append({col: labels.get(v, v), **{k: r[k] for k in ("n", "gap 10d", "gap 20d", "hit 10d %",
                                                                  "t (by day)")}})
    order = {lab: i for i, lab in enumerate(labels.values())}
    return pd.DataFrame(rows).set_index(col).sort_index(key=lambda ix: ix.map(lambda x: order.get(x, len(order))))


def false_positives(s: pd.DataFrame) -> tuple:
    g = s[(s["config"] == MAIN) & (s["side"] == "laggard") & ~s["shock"]].dropna(subset=["gap_20"])
    rows = []
    for name, x in (("all no-shock laggards", g), ("harness catch-up subset", g[g["harness_catch_up"]])):
        rows.append({"group": name, "n": len(x), "still behind peers after 20d %": (x["gap_20"] < 0).mean() * 100,
                     "fell >5 pt further %": (x["gap_20"] < -5).mean() * 100,
                     "median gap 20d": x["gap_20"].median()})
    by_group = (g.groupby("peer_group")["gap_20"].agg(n="size", still_behind=lambda v: (v < 0).mean() * 100,
                                                       mean_gap_20="mean")
                .rename(columns={"still_behind": "still behind %", "mean_gap_20": "mean gap 20d"}))
    worst = (g[g["harness_catch_up"]].nsmallest(8, "gap_20")
             [["date", "ticker", "peer_group", "lag", "gap_20"]]
             .assign(date=lambda x: x["date"].dt.date.astype(str), lag=lambda x: x["lag"] * 100)
             .rename(columns={"lag": "lag at trigger (pt)", "gap_20": "gap 20d (pt)"}))
    return pd.DataFrame(rows).set_index("group"), by_group, worst


def oos_table(s: pd.DataFrame) -> pd.DataFrame:
    g = s[(s["config"] == MAIN) & ~s["shock"]]
    rows = []
    for (side, hc), x in g.groupby(["side", g["harness_catch_up"] & (g["side"] == "laggard")]):
        name = "harness catch-up" if hc else ("other no-shock laggards" if side == "laggard" else "leaders")
        a, b = x[x["date"] < SPLIT], x[x["date"] >= SPLIT]
        rows.append({"group": name, "n first": len(a), "gap 10d first": a["gap_10"].mean(), "n second": len(b),
                     "gap 10d second": b["gap_10"].mean(),
                     "same sign": np.sign(a["gap_10"].mean()) == np.sign(b["gap_10"].mean())})
    return pd.DataFrame(rows).set_index("group")


def backtest_tables() -> tuple:
    b = pd.read_csv("reports/backtest_strategies.csv")
    b = b[b["strateji"].isin(["B_sektor", "A_teknik"])].copy()
    b["XU100"] = b["toplam_getiri"] - b["xu100_fark"]
    out = b.rename(columns={"strateji": "strategy", "donem": "period", "toplam_getiri": "total return",
                            "kiyas_getiri": "equal-weight B&H", "sharpe": "Sharpe", "max_dusus": "max drawdown",
                            "islem_sayisi": "trades", "xu100_fark": "vs XU100"})
    out["period"] = out["period"].map({"tum": "full", "ilk_yari": "first half", "ikinci_yari": "second half"})
    cols = ["strategy", "period", "total return", "XU100", "vs XU100", "equal-weight B&H", "Sharpe", "max drawdown",
            "trades"]
    out = out[cols].set_index(["strategy", "period"])
    for c in ("total return", "XU100", "vs XU100", "equal-weight B&H", "max drawdown"):
        out[c] = out[c] * 100
    cost = pd.read_csv("reports/backtest_cost_sensitivity.csv").set_index("strateji").loc[["B_sektor", "A_teknik"]] * 100
    cost.columns = [f"one-way cost {float(c.split('_')[1]) * 100:.2f}%" for c in cost.columns]
    return out, cost.rename_axis("strategy")


# ---------- grafikler ----------
def fig_ladder(lad: pd.DataFrame) -> str:
    d = lad["gap 10d"]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    style(ax, "x")
    y = np.arange(len(d))[::-1]
    colors = [C["muted"]] + [C["s1"]] * (len(d) - 2) + [C["s2"]]
    ax.barh(y, d.to_numpy(), color=colors, height=0.6)
    for yi, v, n in zip(y, d.to_numpy(), lad["n"].to_numpy()):
        ax.text(v + 0.02, yi, f"{v:+.2f}" + ("" if np.isnan(n) else f"  (n={int(n)})"), va="center", fontsize=8,
                color=C["text2"])
    ax.set_yticks(y, d.index, fontsize=8.5)
    ax.axvline(0, color=C["text2"], linewidth=0.8)
    ax.set_xlim(0, d.max() * 1.45)
    ax.set_xlabel("mean 10-day return minus peer median, percentage points", fontsize=9, color=C["text2"])
    ax.set_title("Figure 1. No-shock, strong-sector and recovery filters help; the earnings filter does not",
                 loc="left", fontsize=10.5, color=C["text"])
    return save_fig(fig, "02_filter_ladder.png")


def fig_groups(s: pd.DataFrame) -> str:
    g = s[(s["config"] == MAIN) & ~s["shock"]]
    t = g.groupby(["peer_group", "side"])["gap_10"].mean().unstack()[["laggard", "leader"]].sort_values("laggard")
    y = np.arange(len(t))
    fig, ax = plt.subplots(figsize=(8, 4.6))
    style(ax, "x")
    h = 0.38
    ax.barh(y + h / 2, t["laggard"], height=h - 0.04, color=C["s1"], label="laggards (≥5 pt behind peers)")
    ax.barh(y - h / 2, t["leader"], height=h - 0.04, color=C["s2"], label="leaders (≥5 pt ahead of peers)")
    ax.axvline(0, color=C["text2"], linewidth=0.8)
    ax.set_yticks(y, t.index, fontsize=8.5)
    ax.set_xlabel("mean 10-day return minus peer median, pt (no-shock events)", fontsize=9, color=C["text2"])
    ax.set_title("Figure 2. Laggard catch-up differs by sector (positive leader bars = leaders kept leading)", loc="left", fontsize=10.5,
                 color=C["text"])
    ax.legend(loc="lower right", frameon=False, fontsize=8.5, labelcolor=C["text2"])
    return save_fig(fig, "02_sector_groups.png")


# ---------- rapor ----------
def build() -> str:
    notes = load_notes(NOTES_FILE)
    s = load_events()
    raw = pd.read_parquet("reports/sector_events.parquet")
    quake = int(raw["quake_window"].sum())
    base = baseline().loc["gap"]
    groups, warn = build_sector_peer_groups()
    gt = pd.DataFrame([(g, len(m), ", ".join(m)) for g, m in groups.items()],
                      columns=["peer group", "size", "members"]).set_index("peer group")
    m = main_table(s)
    lad = filter_ladder(s, base)
    terc = split_table(s, "tercile", {"alt": "bottom third", "orta": "middle third", "ust": "top third",
                                      "unranked": "statement public, quarter not ranked",
                                      "none": "no statement public yet"})
    strong = split_table(s, "sector_strong", {True: "sector median > 0", False: "sector median ≤ 0"})
    recov = split_table(s, "recovering", {True: "above KAMA or Supertrend up", False: "below KAMA and Supertrend down"})
    macro = split_table(s, "macro_risk", {True: "macro risk on (half weight)", False: "macro risk off"})
    fp, fp_g, worst = false_positives(s)
    oos = oos_table(s)
    bt, cost = backtest_tables()
    f1, f2 = fig_ladder(lad), fig_groups(s)
    hc = lad.iloc[-1]

    md = f"""# Report 2 — Sector Laggards and Catch-Up

_{DISCLAIMER}_

Generated by `python -m src.reports.build_reports --only 2` from committed outputs (no analysis is re-run).

{stamp(SOURCES)}

## 1. Research question

Project spec §11: find stocks that are **materially lagging their own sector while the sector is strong**, and
test whether such laggards historically **caught up with their peers** over 5/10/20 trading days, before
treating any of them as a candidate. The spec also asks to separate structural from temporary laggards, to backtest
the rule and to report false positives.

{note(notes, "summary")}

## 2. Data and method

- **Peer groups** (`config/universe.csv`, Step 1). The 30 study stocks plus a few extra BIST 100 peers used only
  to compute sector medians. Groups smaller than 3 are not used for events.

{md_table(gt)}

- **Lag** = stock's n-day return − median n-day return of its peers (the stock itself excluded).
  Negative = behind its sector.
- **Trigger**: the first day the lag crosses the threshold. Two settings: `lag10_5pct` (10 days, ≥5 pt; the
  harness rule) and `lag20_10pct` (20 days, ≥10 pt). The same stock cannot trigger again within {EVENT_GAP}
  trading days. **Leaders** (≥ threshold *ahead* of peers) are the mirror test.
- **Outcome**: `gap_h` = stock's next-h-day return − peers' median next-h-day return, in percentage points.
  Positive for a laggard = it caught up.
- **Shock** = a daily move of at least ±{LIMIT * 100:.1f}% (close to the ±10% price limit) in the 10 days
  before the trigger. Such lags are usually news-driven, not "temporary".
- Earthquake window excluded ({quake} events). Statistics: day-clustered t-test (one mean per trigger day) and
  Benjamini–Hochberg q.
- Any-day baseline (every eligible stock-day): gap 5d **{base['mean_5']:+.2f}**, 10d **{base['mean_10']:+.2f}**,
  20d **{base['mean_20']:+.2f}** pt. This is slightly positive because peer medians and single-stock means differ
  (skewed returns).

## 3. Main result: laggards vs leaders

**Table 1. Mean gap vs peers (pt) after the trigger, earthquake window excluded**

{md_table(m, fmt={"p (by day)": "{:.3f}", "q BH": "{:.3f}", "hit 10d %": "{:.1f}"})}

{note(notes, "main")}

## 4. Does the harness's catch-up rule help?

The harness labels `POTENTIAL_CATCH_UP_CANDIDATE` only if the stock is a `lag10_5pct` laggard **and** has no
shock, **and** its sector median return is positive, **and** it is recovering (close above KAMA or Supertrend up),
**and** its last visible earnings are not in the bottom third. Table 2 adds these filters one at a time.

**Table 2. Filter ladder (lag10_5pct laggards, gap vs peers in pt)**

{md_table(lad, fmt={"p (by day)": "{:.3f}", "hit 10d %": "{:.1f}", "n": "{:,.0f}"})}

![Figure 1]({f1})

{note(notes, "ladder")}

## 5. Structural vs temporary laggards

The spec asks whether a lag is **structural** (the company is weaker) or **temporary**. We use evidence
available on the trigger day: the latest earnings tercile already public (point-in-time, from KAP publish dates),
whether the sector itself is strong, whether the price is recovering, and the macro risk flag used by the
strategies (XU100 below its 200-day average or a USD/TRY spike). All rows are no-shock `lag10_5pct` laggards.

**Table 3a. By latest public earnings tercile (year-over-year net-income change, across the 30 stocks)**

{md_table(terc, fmt={"hit 10d %": "{:.1f}"})}

**Table 3b. By sector strength**

{md_table(strong, fmt={"hit 10d %": "{:.1f}"})}

**Table 3c. By price recovery**

{md_table(recov, fmt={"hit 10d %": "{:.1f}"})}

**Table 3d. By macro risk flag**

{md_table(macro, fmt={"hit 10d %": "{:.1f}"})}

{note(notes, "structural")}

## 6. By sector

![Figure 2]({f2})

{note(notes, "groups")}

## 7. False positives: laggards that kept lagging

**Table 4. Share of no-shock laggards still behind peers 20 days later**

{md_table(fp, fmt={"still behind peers after 20d %": "{:.1f}", "fell >5 pt further %": "{:.1f}"})}

**Table 5. False positives by sector (no-shock laggards)**

{md_table(fp_g, fmt={"still behind %": "{:.1f}"})}

**Table 6. Worst cases that passed every harness filter**

{md_table(worst, index=False)}

{note(notes, "false_positives")}

## 8. Out-of-sample check (split at {SPLIT.date()})

**Table 7. Mean 10-day gap vs peers (pt), lag10_5pct, no shock**

{md_table(oos)}

{note(notes, "oos")}

## 9. Backtest: sector-catch-up strategy (B_sektor)

Step 9 turned the idea into a long-only weekly strategy. Each week it picks up to 10 stocks that are in an
uptrend (Supertrend up, above the 50-day SMA, no bearish KAMA/cloud state) with no shock, ranking the sector laggards
first. Signals are shifted one day (no look-ahead) and a 0.15% one-way cost is charged. The test period is
after a 250-day warm-up. `A_teknik` (same trend filter, no sector ranking) is shown for comparison.
Returns are in %.

**Table 8. Backtest results**

{md_table(bt, fmt={"total return": "{:.0f}", "XU100": "{:.0f}", "vs XU100": "{:+.0f}",
                   "equal-weight B&H": "{:.0f}", "max drawdown": "{:.0f}"})}

**Table 9. Cost sensitivity (total return %, full period)**

{md_table(cost, default="{:.0f}")}

{note(notes, "backtest")}

## 10. Multiple testing

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
    return str(write("02_sector_catchup.md", md))
