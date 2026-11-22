"""Rapor 3 — Weekday and multi-day pattern research report (PDF Bölüm 12).

Kaynak: reports/weekday_portfolio.csv, weekday_per_stock.csv, streak_patterns.csv (Adım 6) + fiyat önbelleği.
Ek testler (fiyattan saniyeler içinde): medyan, rastgele etiket (permutation) testi, XU100 rejimi, TCMB PPK
Perşembeleri, büyük düşüş günü, Pzt-Çar negatif haftalar, Cuma->Pazartesi, Çarşamba al/Perşembe sat maliyetli
backtest. Bütün p değerleri tek bir Benjamini-Hochberg düzeltmesine girer.
"""
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest.engine import benchmarks, calculate_performance, load_returns, run_backtest
from src.backtest.strategies import weekday_test_weights
from src.data.prices import get_index_history
from src.events.calendar_patterns import WEEKDAYS, _data, pattern_table, welch
from src.reports.common import C, DISCLAIMER, bh, load_notes, md_table, note, save_fig, stamp, style, write

SOURCES = ["reports/weekday_portfolio.csv", "reports/weekday_per_stock.csv", "reports/streak_patterns.csv",
           "reports/macro_panel.csv", "config/policy_meetings.csv"]
NOTES_FILE = "03_weekday_patterns.md"
EN = dict(zip(WEEKDAYS, ["Mon", "Tue", "Wed", "Thu", "Fri"]))
BIG_DOWN = (-0.05, -0.095)          # büyük düşüş günü: -%5 ile limit (-%9,5) arası; limit günleri haber şokudur
N_PERM = 5000


def _p_from_t(t):
    return math.erfc(abs(t) / math.sqrt(2)) if np.isfinite(t) else np.nan


def portfolio() -> tuple:
    d = _data()
    port = d["ret"].mean(axis=1)
    return d, port, pd.Series(port.index.day_of_week, index=port.index), d["valid"]


def weekday_table(port, dow, valid) -> pd.DataFrame:
    w = pd.read_csv("reports/weekday_portfolio.csv").set_index("day")
    rng = np.random.default_rng(0)
    pv, dv = port[valid].to_numpy(), dow[valid].to_numpy()
    xu = get_index_history("XU100")["Close"].reindex(port.index)
    sma = xu.rolling(200).mean()
    up, down = valid & (xu > sma), valid & (xu <= sma) & sma.notna()
    rows = []
    for k, name in enumerate(WEEKDAYS):
        obs = pv[dv == k].mean() - pv[dv != k].mean()
        perm = np.empty(N_PERM)
        for i in range(N_PERM):
            s = rng.permutation(dv)
            perm[i] = pv[s == k].mean() - pv[s != k].mean()
        r = w.loc[name]
        rows.append({"day": EN[name], "n": int(r["n"]), "mean %": r["mean"], "median %": np.median(pv[dv == k]) * 100,
                     "diff vs other days": r["diff"], "t": r["t"], "p": r["p"],
                     "p (permutation)": float(np.mean(np.abs(perm) >= abs(obs))),
                     "diff first half": r["diff_IS"], "diff second half": r["diff_OOS"],
                     "mean if XU100 up": port[up & (dow == k)].mean() * 100,
                     "mean if XU100 down": port[down & (dow == k)].mean() * 100})
    return pd.DataFrame(rows).set_index("day")


# ---------- Adım 6 (22 Kasım 2026, sonuçlara bakmadan yazıldı): sektör, sektör yönü, makro haftaları ----------
def _day_vs_rest(series: pd.Series, dow: pd.Series, valid: pd.Series, k: int) -> tuple:
    a, b = series[valid & (dow == k)].dropna(), series[valid & (dow != k)].dropna()
    t, p = welch(a, b)
    return (a.mean() - b.mean()) * 100, t, p, len(a)


def sector_returns_all(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Akran grubu başına eşit ağırlıklı günlük getiri (evrendeki tüm hisseler; backtest sektör kıyasıyla aynı)."""
    from src.backtest.engine import sector_returns
    return sector_returns(index)[0]


def sector_weekday(dow, valid) -> pd.DataFrame:
    """10 sektör portföyü x 5 gün: o gün diğer günlerden farklı mı (50 test)."""
    sec = sector_returns_all(dow.index)
    rows = []
    for g in sec.columns:
        for k, name in enumerate(WEEKDAYS):
            diff, t, p, n = _day_vs_rest(sec[g], dow, valid, k)
            rows.append({"sector": g, "day": EN[name], "n": n, "diff vs other days %": diff, "t": t, "p": p})
    return pd.DataFrame(rows)


def sector_direction(d, dow, valid) -> pd.DataFrame:
    """R3'ün iki bulgusu (Çarşamba zayıf, Perşembe güçlü), hissenin sektörü önceki 5 günde yükselmişken ve
    düşmüşken ayrı ayrı. Sektör yönü t-5..t-1 getirisi (t günü açılışında bilinir). 4 test."""
    from src.data.universe import load_universe
    grp = load_universe().set_index("ticker")["peer_group"]
    sec = sector_returns_all(d["ret"].index)
    prior = (1 + sec).rolling(5).apply(np.prod, raw=True).sub(1).shift(1)
    sec_dir = pd.DataFrame({t: prior[grp[t]] for t in d["ret"].columns})
    rows = []
    for lab, cond in (("sector up in previous 5 days", sec_dir > 0), ("sector down in previous 5 days", sec_dir < 0)):
        port = d["ret"].where(cond).mean(axis=1)
        for k in (2, 3):
            diff, t, p, n = _day_vs_rest(port, dow, valid, k)
            rows.append({"condition": lab, "day": EN[WEEKDAYS[k]], "days": n, "diff vs other days %": diff, "t": t, "p": p})
    return pd.DataFrame(rows)


def macro_weeks(idx: pd.DatetimeIndex) -> pd.DataFrame:
    """Her işlem günü için: haftasında TCMB PPK kararı / Fed FOMC kararı / TÜFE açıklaması var mı.
    TCMB 14:00'te açıklar (aynı gün); Fed BIST kapandıktan sonra açıklar (sonraki işlem günü); TÜFE ayın 3'ü."""
    from src.data.macro import cpi_release_date
    m = pd.read_csv("config/policy_meetings.csv", parse_dates=["date"])
    def first_on_or_after(days):
        pos = idx.searchsorted(pd.DatetimeIndex(days))
        return idx[pos[pos < len(idx)]]
    ev = {"TCMB decision week": first_on_or_after(m.loc[m["bank"] == "TCMB", "date"]),
          "Fed decision week": first_on_or_after(m.loc[m["bank"] == "FED", "date"] + pd.Timedelta(days=1)),
          "CPI release week": first_on_or_after([cpi_release_date(x) for x in
                                                 pd.date_range(idx[0] - pd.offsets.MonthBegin(2), idx[-1], freq="MS")])}
    wk = idx.to_period("W")
    return pd.DataFrame({k: wk.isin(pd.DatetimeIndex(v).to_period("W")) for k, v in ev.items()}, index=idx)


def macro_week_tests(port, dow, valid) -> pd.DataFrame:
    """Her makro hafta türü için: o haftaların Çarşambası / Perşembesi diğer haftalarınkinden farklı mı, hafta
    toplamı (geçerli günlerin birleşik getirisi) diğer haftalardan farklı mı. 3 x 3 = 9 test."""
    mw = macro_weeks(port.index)
    wk = port.index.to_period("W")
    weekly = port[valid].groupby(wk[valid.to_numpy()]).apply(lambda x: (1 + x).prod() - 1)
    rows = []
    for typ in mw.columns:
        e = mw[typ]
        for k in (2, 3):
            a, b = port[valid & e & (dow == k)], port[valid & ~e & (dow == k)]
            t, p = welch(a, b)
            rows.append({"week type": typ, "test": f"{EN[WEEKDAYS[k]]} in these weeks vs other weeks", "n": len(a),
                         "effect %": (a.mean() - b.mean()) * 100, "t": t, "p": p})
        ew = pd.Series(e.to_numpy(), index=wk).groupby(level=0).any().reindex(weekly.index)
        t, p = welch(weekly[ew], weekly[~ew])
        rows.append({"week type": typ, "test": "whole-week return vs other weeks", "n": int(ew.sum()),
                     "effect %": (weekly[ew].mean() - weekly[~ew].mean()) * 100, "t": t, "p": p})
    return pd.DataFrame(rows)


def per_stock() -> tuple:
    s = pd.read_csv("reports/weekday_per_stock.csv")
    s["day"] = s["day"].map(EN)
    counts = pd.DataFrame([{"tests": len(s), "raw p < 0.05": int((s["p"] < 0.05).sum()),
                            "expected by chance": 0.05 * len(s), "BH q < 0.05": int((s["q_bh"] < 0.05).sum()),
                            "Bonferroni p < 0.05": int((s["p_bonf"] < 0.05).sum())}])
    top = s.sort_values("p").head(8).rename(columns={"diff": "diff vs other days %", "q_bh": "q BH (150 tests)",
                                                      "p_bonf": "p Bonferroni"})
    opp = []
    for day, g in s.groupby("day"):
        hi, lo = g.loc[g["diff"].idxmax()], g.loc[g["diff"].idxmin()]
        opp.append({"day": day, "strongest stock": f"{hi['ticker']} ({hi['diff']:+.2f})",
                    "weakest stock": f"{lo['ticker']} ({lo['diff']:+.2f})",
                    "stocks above 0": int((g["diff"] > 0).sum())})
    order = [EN[w] for w in WEEKDAYS]
    opp = pd.DataFrame(opp).set_index("day").loc[order]
    return counts, top, opp, s


def ppk_thursdays(port, dow, valid) -> tuple:
    m = pd.read_csv("reports/macro_panel.csv", parse_dates=["date"])
    ch = m.loc[m["policy_rate"].diff().fillna(0) != 0, "date"]
    ch = ch[ch > m["date"].iloc[0]]
    is_ch = pd.Series(port.index.isin(ch), index=port.index)
    thu = valid & (dow == 3)
    others = valid & (dow != 3)
    a, b, c = port[thu & is_ch], port[thu & ~is_ch], port[others]
    t, p = welch(b, c)
    t2, p2 = welch(a, b)
    tab = pd.DataFrame([
        {"group": "Thursdays with a policy-rate change", "n": len(a), "mean %": a.mean() * 100},
        {"group": "other Thursdays", "n": len(b), "mean %": b.mean() * 100},
        {"group": "all non-Thursdays", "n": len(c), "mean %": c.mean() * 100},
    ]).set_index("group")
    all_thu_wd = sum(pd.Timestamp(x).day_of_week == 3 for x in ch)
    return tab, dict(t_other=t, p_other=p, t_change=t2, p_change=p2, n_changes=len(ch), on_thu=all_thu_wd)


def new_patterns(d, port, dow, valid) -> pd.DataFrame:
    ret, panel = d["ret"], d["panel"]
    rows = []
    # büyük düşüş günü -> aynı günün hisse ortalamasına göre fark (gün kümelemeli t)
    m = (ret <= BIG_DOWN[0]) & (ret > BIG_DOWN[1]) & valid.to_numpy()[:, None]
    for h in (1, 3, 5):
        fwd = panel.shift(-h) / panel - 1
        bd = fwd.sub(fwd.mean(axis=1), axis=0).where(m).mean(axis=1).dropna()
        t = bd.mean() / (bd.std(ddof=1) / np.sqrt(len(bd)))
        rows.append({"pattern": f"stock-day ≤ −5% (not limit) → next {h}d vs same-day average", "events": int(m.sum().sum()),
                     "dates/weeks": len(bd), "effect %": bd.mean() * 100, "t": t, "p": _p_from_t(t)})
    # üç ardışık düşüş -> iki günlük toparlanma (spec örneği)
    pt = pattern_table(ret, panel, valid, ks=(3,), horizons=(2,)).loc[("down3", 2)]
    rows.append({"pattern": "3 down days in a row → next 2d vs same-day average", "events": int(pt["n_events"]),
                 "dates/weeks": int(pt["n_dates"]), "effect %": pt["mean_ab"], "t": pt["t"], "p": pt["p"]})
    # haftalık: Pzt-Sal-Çar negatif -> Per+Cum ; Cuma yönü -> Pazartesi
    wk = pd.DataFrame({"r": port, "dow": dow, "wk": port.index.to_period("W")})[valid]
    g = wk.pivot_table(index="wk", columns="dow", values="r")
    full = g.dropna(subset=[0, 1, 2, 3, 4])
    cond = (full[0] < 0) & (full[1] < 0) & (full[2] < 0)
    tf = (1 + full[3]) * (1 + full[4]) - 1
    t, p = welch(tf[cond], tf[~cond])
    rows.append({"pattern": "Mon, Tue, Wed all negative → Thu+Fri (vs other weeks)", "events": int(cond.sum()),
                 "dates/weeks": int(cond.sum()), "effect %": (tf[cond].mean() - tf[~cond].mean()) * 100, "t": t, "p": p})
    fr, mo = g[4], g[0].shift(-1)
    ok = fr.notna() & mo.notna()
    t, p = welch(mo[ok & (fr > 0)], mo[ok & (fr < 0)])
    rows.append({"pattern": "next Monday after an up Friday minus after a down Friday", "events": int(ok.sum()),
                 "dates/weeks": int(ok.sum()), "effect %": (mo[ok & (fr > 0)].mean() - mo[ok & (fr < 0)].mean()) * 100,
                 "t": t, "p": p})
    return pd.DataFrame(rows).set_index("pattern")


def wed_thu_backtest() -> tuple:
    ret = load_returns()
    w = weekday_test_weights()
    bm = benchmarks(ret)
    rows, curves = [], {}
    for cost in (0.0, 0.0015, 0.003):
        bt = run_backtest(w.reindex(ret.index).fillna(0.0), ret, cost=cost)
        p = calculate_performance(bt, bm["xu100"])
        rows.append({"one-way cost %": cost * 100, "total return %": p["toplam_getiri"] * 100,
                     "XU100 %": p["kiyas_getiri"] * 100, "Sharpe": p["sharpe"], "max drawdown %": p["max_dusus"] * 100,
                     "days in market %": p["piyasada_gun"] * 100, "trades": p["islem_sayisi"],
                     "total cost (sum of daily %)": p["toplam_maliyet"] * 100})
        curves[f"Wed→Thu, cost {cost * 100:.2f}%"] = (1 + bt["net"].fillna(0)).cumprod()
    curves["XU100"] = (1 + bm["xu100"].fillna(0)).cumprod()
    return pd.DataFrame(rows).set_index("one-way cost %"), pd.DataFrame(curves)


# ---------- grafikler ----------
def fig_is_oos(wt: pd.DataFrame) -> str:
    x = np.arange(len(wt))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    style(ax, "y")
    b = 0.38
    ax.bar(x - b / 2, wt["diff first half"], width=b - 0.04, color=C["s1"], label="first half")
    ax.bar(x + b / 2, wt["diff second half"], width=b - 0.04, color=C["s2"], label="second half")
    ax.axhline(0, color=C["text2"], linewidth=0.8)
    ax.set_xticks(x, wt.index)
    ax.set_ylabel("day's mean minus other days, %", fontsize=9, color=C["text2"])
    ax.set_title("Figure 1. Only Thursday keeps its sign and size in the second half", loc="left", fontsize=10.5,
                 color=C["text"])
    ax.legend(frameon=False, fontsize=8.5, labelcolor=C["text2"])
    return save_fig(fig, "03_weekday_is_oos.png")


def fig_costs(curves: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4))
    style(ax, "y")
    cols = {curves.columns[0]: C["s1"], curves.columns[1]: C["s2"], curves.columns[2]: "#1baf7a", "XU100": C["muted"]}
    for c, col in cols.items():
        ax.plot(curves.index, curves[c], color=col, linewidth=2 if c != "XU100" else 1.5,
                linestyle="--" if c == "XU100" else "-", label=c)
        ax.text(curves.index[-1], curves[c].iloc[-1], f"  {curves[c].iloc[-1]:.1f}×", va="center", fontsize=8,
                color=C["text2"])
    ax.set_yscale("log")
    ax.set_ylabel("growth of 1 TRY (log scale)", fontsize=9, color=C["text2"])
    ax.set_title("Figure 2. Wednesday-buy / Thursday-sell: the edge disappears once trading costs are paid",
                 loc="left", fontsize=10.5, color=C["text"])
    ax.legend(frameon=False, fontsize=8.5, labelcolor=C["text2"], loc="upper left")
    return save_fig(fig, "03_wed_thu_costs.png")


# ---------- rapor ----------
def build() -> str:
    notes = load_notes(NOTES_FILE)
    d, port, dow, valid = portfolio()
    idx = port.index
    half = idx[len(idx) // 2]
    wt = weekday_table(port, dow, valid)
    counts, top, opp, per = per_stock()
    ppk, ppk_s = ppk_thursdays(port, dow, valid)
    streak = pd.read_csv("reports/streak_patterns.csv").set_index(["pattern", "h"])
    newp = new_patterns(d, port, dow, valid)
    secw = sector_weekday(dow, valid)
    secd = sector_direction(d, dow, valid)
    mwt = macro_week_tests(port, dow, valid)
    mw_counts = macro_weeks(port.index).groupby(port.index.to_period("W")).any().sum()
    bt, curves = wed_thu_backtest()
    f1, f2 = fig_is_oos(wt), fig_costs(curves)

    # tek BH düzeltmesi: bu rapordaki bütün hipotez testleri
    fam = pd.concat([
        pd.DataFrame({"family": "weekday (portfolio)", "test": wt.index, "p": wt["p"].to_numpy()}),
        pd.DataFrame({"family": "weekday (per stock)", "test": per["ticker"] + " " + per["day"], "p": per["p"]}),
        pd.DataFrame({"family": "streaks", "test": [f"{a} h={b}" for a, b in streak.index], "p": streak["p"].to_numpy()}),
        pd.DataFrame({"family": "new patterns", "test": newp.index, "p": newp["p"].to_numpy()}),
        pd.DataFrame({"family": "PPK Thursdays", "test": ["other Thursdays vs non-Thursdays"], "p": [ppk_s["p_other"]]}),
        pd.DataFrame({"family": "sector portfolios (pre-registered)", "test": secw["sector"] + " " + secw["day"],
                      "p": secw["p"]}),
        pd.DataFrame({"family": "sector direction (pre-registered)", "test": secd["condition"] + " / " + secd["day"],
                      "p": secd["p"]}),
        pd.DataFrame({"family": "macro-event weeks (pre-registered)", "test": mwt["week type"] + " / " + mwt["test"],
                      "p": mwt["p"]}),
    ], ignore_index=True)
    fam["q (all tests)"] = bh(fam["p"])
    fam_sum = fam.groupby("family", sort=False).agg(tests=("p", "size"), raw_p=("p", lambda x: int((x < 0.05).sum())),
                                                    q=("q (all tests)", lambda x: int((x < 0.05).sum())))
    fam_sum.columns = ["tests", "raw p < 0.05", "q < 0.05 (all tests together)"]
    survivors = fam[fam["q (all tests)"] < 0.05].sort_values("p")[["family", "test", "p", "q (all tests)"]]
    newp["q (all tests)"] = fam.loc[fam["family"] == "new patterns", "q (all tests)"].to_numpy()
    for tab, famname in ((secw, "sector portfolios (pre-registered)"), (secd, "sector direction (pre-registered)"),
                         (mwt, "macro-event weeks (pre-registered)")):
        tab["q (all tests)"] = fam.loc[fam["family"] == famname, "q (all tests)"].to_numpy()
    sec_piv = secw.pivot(index="sector", columns="day", values="diff vs other days %")[[EN[w] for w in WEEKDAYS]]
    sec_low = secw[secw["p"] < 0.05].sort_values("p")[["sector", "day", "diff vs other days %", "p", "q (all tests)"]]
    streak_v = streak.reset_index()
    streak_v["q (all tests)"] = fam.loc[fam["family"] == "streaks", "q (all tests)"].to_numpy()
    streak_v = streak_v.rename(columns={"mean_ab": "mean %", "IS": "first half %", "OOS": "second half %",
                                        "q_bh": "q (18 tests)"}).drop(columns=["p_bonf"]).set_index(["pattern", "h"])

    md = f"""# Report 3 — Weekday and Multi-Day Patterns

_{DISCLAIMER}_

Generated by `python -m src.reports.build_reports --only 3` from committed outputs plus the cached price data.

{stamp(SOURCES)}

## 1. Research question

Project spec §12: do the 30 stocks, or the market they form, show **recurring day-of-week or multi-day
sequences**, and do those patterns survive regimes, unseen later periods, comparison with random labels,
transaction costs and multiple testing? The spec's warning is the point of the exercise: searching many
combinations will produce patterns by chance.

{note(notes, "summary")}

## 2. Data and method

- Daily close-to-close returns of the 30 study stocks, {idx[1].date()} – {idx[-1].date()}. "Portfolio" = the
  equal-weight average of the 30 stocks on each day.
- **Valid days only**: a day directly after a holiday (or a Monday after a long weekend) is excluded, because its
  return covers more than one trading session. The earthquake closure (Feb 2023) is excluded.
  {int(valid.sum())} of {len(valid)} days are valid.
- Weekday tests compare one weekday with all other weekdays (Welch t-test). A **permutation test** reshuffles the
  weekday labels {N_PERM:,} times: how often does a random "weekday" look as extreme?
- Multi-day patterns are measured as **abnormal return**: the stock's next-h-day return minus the average of all
  30 stocks over the same days. Market moves are removed, and each date counts once.
- First half / second half split at **{half.date()}**.
- Every p-value in this report (portfolio, 150 per-stock, 18 streak, 6 new patterns, 1 PPK test) is corrected
  together with Benjamini–Hochberg (Section 8).

## 3. Day of week (30-stock portfolio)

**Table 1. Portfolio return by weekday (%)**

{md_table(wt, fmt={"p": "{:.4f}", "p (permutation)": "{:.4f}", "mean %": "{:.3f}", "median %": "{:.3f}",
                   "diff vs other days": "{:+.3f}", "diff first half": "{:+.3f}", "diff second half": "{:+.3f}",
                   "mean if XU100 up": "{:+.3f}", "mean if XU100 down": "{:+.3f}"})}

![Figure 1]({f1})

{note(notes, "weekday")}

### 3.1 Thursdays and central-bank decisions

All {ppk_s['n_changes']} policy-rate changes in our TCMB data fall on a Thursday ({ppk_s['on_thu']} of
{ppk_s['n_changes']}): the Monetary Policy Committee announces its decision on Thursdays at 14:00.

**Table 2. Thursday returns with and without a rate change (%)**

{md_table(ppk, fmt={"mean %": "{:+.3f}"})}

Other Thursdays vs all non-Thursdays: t = {ppk_s['t_other']:.2f}, p = {ppk_s['p_other']:.3f}.
Rate-change Thursdays vs other Thursdays: t = {ppk_s['t_change']:.2f}, p = {ppk_s['p_change']:.3f}.

{note(notes, "ppk")}

## 4. Day of week, stock by stock

**Table 3. 150 tests (30 stocks × 5 days)**

{md_table(counts, index=False, fmt={"expected by chance": "{:.1f}"})}

**Table 4. Strongest single-stock results**

{md_table(top, index=False, fmt={"diff vs other days %": "{:+.3f}", "t": "{:.2f}", "p": "{:.4f}",
                                 "q BH (150 tests)": "{:.3f}", "p Bonferroni": "{:.3f}"})}

**Table 5. The same weekday helps some stocks and hurts others**

{md_table(opp)}

{note(notes, "per_stock")}

## 5. Streaks: consecutive up or down days

**Table 6. Abnormal return after k consecutive down/up days (% vs same-day 30-stock average)**

{md_table(streak_v, fmt={"mean %": "{:+.3f}", "t": "{:.2f}", "p": "{:.3f}", "first half %": "{:+.3f}",
                         "second half %": "{:+.3f}", "q (18 tests)": "{:.3f}", "q (all tests)": "{:.3f}"})}

{note(notes, "streaks")}

## 6. Patterns named in the spec

**Table 7. Large down days, three-down-day rebound, the Mon–Wed sequence and Friday → Monday**

{md_table(newp, fmt={"effect %": "{:+.3f}", "t": "{:.2f}", "p": "{:.4f}", "q (all tests)": "{:.3f}"})}

{note(notes, "new")}

## 6b. Sectors, sector direction and macro-event weeks (pre-registered)

The spec asks for patterns in **sectors** and **conditional on sector direction or macro-event weeks**. These
tests were written on 22 November 2026, before their results were computed, and join the report's single
Benjamini–Hochberg family.

- **Sector portfolios.** Ten equal-weight sector portfolios (all universe members of the sector), each weekday vs
  the other days of the same portfolio: {len(secw)} tests.
- **Sector direction.** The two portfolio findings of Section 3 (weak Wednesday, strong Thursday), computed
  separately on stock-days whose sector rose / fell over the previous five trading days (known before the day):
  {len(secd)} tests.
- **Macro-event weeks.** Weeks with a TCMB rate decision (announced 14:00, same day), a Fed FOMC decision (announced
  after the Borsa İstanbul close, so the next trading day counts) or a CPI release (TÜİK, the 3rd of the month).
  The meeting dates, including meetings without a rate change, are in `config/policy_meetings.csv` with the
  official source of each: {", ".join(f"{k} {int(v)}" for k, v in mw_counts.items())} weeks. For each type:
  Wednesday and Thursday in these weeks vs the same day in other weeks, and the whole-week return: {len(mwt)} tests.

**Table 7a. Sector portfolios: weekday minus other days (%, mean daily return)**

{md_table(sec_piv, default="{:+.3f}")}

Sector-day cells with p < 0.05 (before correction):

{md_table(sec_low, index=False, fmt={"diff vs other days %": "{:+.3f}", "p": "{:.4f}", "q (all tests)": "{:.3f}"})
 if len(sec_low) else "none"}

**Table 7b. Wednesday and Thursday by the sector's direction over the previous five days**

{md_table(secd, index=False, fmt={"diff vs other days %": "{:+.3f}", "t": "{:.2f}", "p": "{:.4f}", "q (all tests)": "{:.3f}"})}

**Table 7c. Macro-event weeks**

{md_table(mwt, index=False, fmt={"effect %": "{:+.3f}", "t": "{:.2f}", "p": "{:.4f}", "q (all tests)": "{:.3f}"})}

{note(notes, "sector_macro")}

## 7. Transaction costs: trading the best calendar pattern

The strongest weekday result is Wednesday-weak / Thursday-strong. The simplest way to trade it: buy all 30
stocks equally at Wednesday's close and sell at Thursday's close, every week (signal shifted one day, Step 9 engine).

**Table 8. Wednesday-buy / Thursday-sell backtest**

{md_table(bt, fmt={"total return %": "{:.0f}", "XU100 %": "{:.0f}", "max drawdown %": "{:.0f}",
                   "days in market %": "{:.0f}", "total cost (sum of daily %)": "{:.0f}"})}

![Figure 2]({f2})

{note(notes, "costs")}

## 8. Multiple testing

**Table 9. How many tests were run and how many survive**

{md_table(fam_sum)}

**Table 10. Results with q < 0.05 after correcting all {len(fam)} tests together**

{md_table(survivors, index=False, fmt={"p": "{:.5f}", "q (all tests)": "{:.3f}"})}

{note(notes, "multiple")}

## 9. Türkiye context

{note(notes, "turkey")}

## 10. How the harness uses these results

{note(notes, "harness")}

## 11. Limitations

{note(notes, "limitations")}

## 12. Conclusion

{note(notes, "conclusion")}
"""
    return str(write("03_weekday_patterns.md", md))
