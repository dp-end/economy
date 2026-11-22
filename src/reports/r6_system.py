"""Rapor 6 — System report: backtests, MCP tools + stateful harness, harness experiments A–E, decision log/replay.

Kaynak: reports/backtest_*.csv (Adım 9-10), MCP sunucusu (Adım 12), durum makinesi ve kapı (Adım 11-13),
reports/harness_experiments.csv + experiments/per_case.csv (Adım 14), reports/decision_log.sqlite (Adım 11-14).
Replay ve backtest eğrileri bu rapor üretilirken yeniden hesaplanır (anahtar gerekmez).
"""
import asyncio
import json
import sqlite3

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mcp import Client

from src.backtest.engine import (COST_ONE_WAY, benchmarks, calculate_performance, load_returns, run_backtest)
from src.backtest.strategies import TOP_K, WARMUP, generate_strategy_variants
from src.harness import quality_gate as G
from src.harness.decision_log import DB, PROCEDURE, replay_analysis
from src.harness.evidence import ACTIONS
from src.harness.experiments import CASES
from src.harness.state_machine import AGENT_AT, MANDATORY, SAFE_PATH, STATE_TOOLS, STATES, VERSION_STATES
from src.mcp_server.server import build_server
from src.reports.common import C, DISCLAIMER, load_notes, md_table, note, save_fig, stamp, style, write

SOURCES = ["reports/backtest_strategies.csv", "reports/backtest_cost_sensitivity.csv", "reports/backtest_trades.csv",
           "reports/backtest_rolling.csv", "reports/backtest_walkforward.csv",
           "reports/backtest_walkforward_summary.csv",
           "reports/harness_experiments.csv", "reports/experiments/per_case.csv", "reports/experiments/raw.jsonl",
           "reports/decision_log.sqlite", "reports/scans/scan_2026-01-09_POTENTIAL_CATCH_UP_CANDIDATE.csv",
           "reports/scans/scan_2026-11-06_POTENTIAL_CATCH_UP_CANDIDATE.csv"]
SCAN_FILES = SOURCES[-2:]
SCAN_STEPS = [  # PDF §33 adımı -> sistemdeki karşılığı
    ("1 Orchestrator selects the 30-stock universe", "`scan_universe` loops over the core universe"),
    ("2 Market tool loads OHLCV and XU100", "`LOAD_MARKET_DATA` / evidence bundle (prices up to the scan day)"),
    ("3 Data-quality check", "quality gate per stock (`gate` column); full harness repeats it in `CHECK_DATA_QUALITY`"),
    ("4 Sector tool ranks each stock vs peers", "`sector_lag_10`, `sector_median_10` (`rank_sector_relative_strength`)"),
    ("5 Technical states and reversal history", "`recovering`, `technical`, `tech_conflict`; event states (Report 1)"),
    ("6 Point-in-time fundamentals", "`weak_earnings`, `fundamental` (statements public by the scan day)"),
    ("7 Macro context", "full harness `LOAD_MACRO_CONTEXT` (selected candidates)"),
    ("8 News / speech events", "`negative_news`; full harness `LOAD_NEWS_AND_SPEECH_CONTEXT`"),
    ("9 Evidence builder", "`run_analysis` per stock: observed vs committed evidence"),
    ("10–11 Strategy variants and backtests", "full harness `PREVIEW_STRATEGIES`, `RUN_BACKTEST` (data before the day)"),
    ("12 Risk gate", "gate per stock; candidates with shock / negative news become `REJECT_SIGNAL`"),
    ("13 Analytical conclusion", "Orchestrator explanation for each selected candidate"),
    ("14 Human review", "`HUMAN_REVIEW` for each selected candidate"),
    ("15 Saved for replay", "`scans` table (+ `analyses` of the candidates), `replay_scan`"),
]
NOTES_FILE = "06_system_harness.md"
VARIANTS = {
    "A_teknik": ("A — technical only", "Supertrend up, close > SMA50, no bearish KAMA slope or below-cloud state; "
                 "ranked by 20-day return vs XU100"),
    "B_sektor": ("B — + sector relative", "A without shock stocks; sector laggards (bottom third of 10-day lag) "
                 "ranked first"),
    "C_temel": ("C — + fundamentals", "B without stocks whose latest public earnings are in the bottom third"),
    "D_makro": ("D — + macro", "C at half size while the macro risk flag is on"),
    "D_makro_oynaklik": ("D variant", "D with inverse-volatility weights"),
    "D_makro_faiz": ("D variant", "D, also half size during rate-hiking periods"),
    "E_haber": ("E — + news", "D without stocks with negative KAP news in 10 days; positive news ranked up"),
}
TOOL_EN = {
    "get_stock_history": "daily High/Low/Close/Volume/turnover with source metadata",
    "get_index_history": "XU100 benchmark series",
    "calculate_indicators": "SMA/EMA/RSI/MACD/Bollinger/ATR/KAMA/Supertrend/Ichimoku, past data only",
    "detect_reversal_events": "timestamped technical events with the indicator state at the event; forward returns only if their window closed by `end`; `event_set=harness` returns the 12 types the harness uses",
    "find_support_resistance": "nearest confirmed support/resistance, touch count and age on a date",
    "rank_sector_relative_strength": "peer returns, lag vs peer median (self excluded), rank",
    "test_weekday_pattern": "weekday or streak pattern vs unconditional returns, half-sample stability, cost note",
    "get_fundamental_metrics": "latest statement public on `asof`, with publication time",
    "calculate_fundamental_change": "QoQ / YoY changes of statements visible on `asof`",
    "get_macro_context": "FX, policy rate, CPI, Fed range as published on each day",
    "get_news_events": "KAP/RSS headline metadata, source link and timestamp",
    "get_speech_features": "topics, stance, uncertainty, verified vs unverified claims of a speech",
    "run_backtest": "strategy variant performance (1-day signal shift, costs) and trade summary",
    "check_data_quality": "quality gate: hard blocks, unsafe states, soft warnings",
    "build_evidence_bundle": "observed context + committed evidence + rule action + gate (or read from the log)",
}
AGENT_EN = {"market": "Market & Relative-Value", "fundamental_macro": "Fundamental & Macro",
            "news_speech": "News & Speech Intelligence", "strategy": "Strategy & Backtest",
            "risk": "Risk & Data Quality", "orchestrator": "Financial Orchestrator"}
METRIC_EN = {"vaka_tekrar": "case runs", "gecerli_eylem": "valid action", "referansla_ayni": "same as reference",
             "desteksiz_sonuc": "unsupported conclusion", "tekrarlanabilirlik": "reproducibility (run 1 = run 2)",
             "sayisal_halusinasyon": "numerical hallucination", "yuvarlama_hatasi": "rounding error",
             "kanit_tamligi": "evidence completeness", "gecersiz_arac_orani": "invalid / failed tool calls",
             "gelecege_bakan_cagri": "look-ahead tool calls", "backtest_yapildi": "backtest run",
             "arac_cagrisi": "tool calls per run", "llm_cagrisi": "LLM calls per run", "sure_sn": "seconds per run"}
CASE_EN = {"bilanço adayı": "earnings candidate", "şok + olumsuz haber": "shock + negative news",
           "çelişen teknik sinyaller": "conflicting technical signals", "şok nedeniyle ret": "rejected because of a shock",
           "deprem penceresi, analiz güvensiz": "earthquake window, analysis unsafe",
           "bilinmeyen hisse": "unknown ticker"}
VERSION_EN = {"A": "raw LLM: data as text, no tools", "B": "LLM + 15 MCP tools, free use",
              "C": "tools + state machine (ordered states, per-state tool permissions); LLM picks the action",
              "D": "C + evidence bundle + quality gate; rules pick the action, LLM explains",
              "E": "D + episodic memory + human review + decision log (full system)"}


# ---------- 1. backtest ----------
def backtest_tables() -> tuple:
    b = pd.read_csv("reports/backtest_strategies.csv")
    b["xu100_total"] = b["toplam_getiri"] - b["xu100_fark"]
    t = pd.DataFrame({
        "strategy": b["strateji"], "period": b["donem"].map({"tum": "full", "ilk_yari": "first half",
                                                             "ikinci_yari": "second half"}),
        "total %": b["toplam_getiri"] * 100, "annual %": b["yillik_getiri"] * 100, "vol %": b["yillik_oynaklik"] * 100,
        "Sharpe": b["sharpe"], "max DD %": b["max_dusus"] * 100, "win days %": b["kazanan_gun"] * 100,
        "in market %": b["piyasada_gun"] * 100, "entries": b["islem_sayisi"], "costs %": b["toplam_maliyet"] * 100,
        "vs equal-weight pt": b["fark"] * 100, "vs XU100 pt": b["xu100_fark"] * 100, "info ratio": b["bilgi_orani"],
        "flag": b["uyari"].astype(object).where(b["uyari"].notna(), "")})
    if not t["flag"].astype(bool).any():                     # hiçbir varyantta "az sinyal" uyarısı yok
        t = t.drop(columns="flag")
    bench = (b[b["strateji"] == "A_teknik"][["donem", "kiyas_getiri", "xu100_total"]]
             .assign(period=lambda x: x["donem"].map({"tum": "full", "ilk_yari": "first half",
                                                      "ikinci_yari": "second half"}))
             .set_index("period")[["kiyas_getiri", "xu100_total"]] * 100)
    bench.columns = ["equal-weight buy & hold % (no cost)", "XU100 %"]
    cost = pd.read_csv("reports/backtest_cost_sensitivity.csv").set_index("strateji") * 100
    cost.columns = [f"one-way cost {float(c.split('_')[1]) * 100:.2f}%" for c in cost.columns]
    return t.set_index(["strategy", "period"]), bench, cost.rename_axis("strategy")


def backtest_extra() -> tuple:
    """Adım 5: sektör nötr kıyas + işlem bazında sonuçlar, kayan pencereler, walk-forward seçim."""
    b = pd.read_csv("reports/backtest_strategies.csv")
    per = {"tum": "full", "ilk_yari": "first half", "ikinci_yari": "second half"}
    sec = pd.DataFrame({"strategy": b["strateji"], "period": b["donem"].map(per),
                        "total %": b["toplam_getiri"] * 100, "sector-neutral %": b["sektor_kiyas_getiri"] * 100,
                        "vs sector-neutral pt": b["sektor_farki"] * 100, "closed trades": b["islem_kapanan"],
                        "winning trades %": b["islem_kazanma_orani"] * 100,
                        "median trade %": b["islem_ortanca_getiri"] * 100,
                        "trades beating XU100 %": b["islem_xu100_gecme_orani"] * 100,
                        "avg days held": b["islem_ort_gun"]}).set_index(["strategy", "period"])
    r = pd.read_csv("reports/backtest_rolling.csv")
    n = r["pencere"].nunique()
    roll = r.groupby("strateji").agg(**{f"windows beating XU100 (of {n})": ("fark_xu100", lambda x: int((x > 0).sum())),
                                       f"beating equal weight (of {n})": ("fark_esit_agirlik", lambda x: int((x > 0).sum())),
                                       f"beating sector-neutral (of {n})": ("fark_sektor_notr", lambda x: int((x > 0).sum())),
                                       "worst window vs XU100 pt": ("fark_xu100", lambda x: x.min() * 100),
                                       "best window vs XU100 pt": ("fark_xu100", lambda x: x.max() * 100)})
    wf = pd.read_csv("reports/backtest_walkforward.csv")
    wf = pd.DataFrame({"test window": wf["bas"] + " – " + wf["bit"], "chosen (best Sharpe, previous 12 months)": wf["secilen"],
                       "training Sharpe": wf["egitim_sharpe"], "chosen return %": wf["secilen_getiri"] * 100,
                       "XU100 %": wf["xu100"] * 100, "best in hindsight": wf["en_iyi_sonradan"]})
    ws = pd.read_csv("reports/backtest_walkforward_summary.csv")
    ws = pd.DataFrame({"method": ws["yontem"].str.replace("walk-forward secim", "walk-forward selection")
                       .str.replace("sabit ", "always ").str.replace("kiyas esit_agirlik", "equal-weight buy & hold")
                       .str.replace("kiyas xu100", "XU100"),
                       "total %": ws["toplam_getiri"] * 100, "Sharpe": ws["sharpe"], "max DD %": ws["max_dusus"] * 100})
    return sec, roll.rename_axis("strategy"), wf, ws


def lookahead_check() -> pd.DataFrame:
    """Kasıtlı hata: bugünün getirisini bilen sinyal. Kaydırma yoksa absürt getiri, 1 gün kaydırınca kaybolur."""
    ret = load_returns()
    bm = benchmarks(ret)
    cheat = (ret > 0).astype(float)
    cheat = cheat.div(cheat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    rows = []
    for name, kw in (("signal uses today's return, no shift (deliberate bug)", dict(shift=0)),
                     ("same signal, shifted 1 day (correct)", dict(shift=1)),
                     ("same signal, shifted, zero cost", dict(shift=1, cost=0.0))):
        p = calculate_performance(run_backtest(cheat, ret, **kw), bm["xu100"])
        rows.append({"test": name, "total %": p["toplam_getiri"] * 100, "Sharpe": p["sharpe"],
                     "max DD %": p["max_dusus"] * 100, "entries": p["islem_sayisi"]})
    return pd.DataFrame(rows).set_index("test")


def equity_curves() -> pd.DataFrame:
    ret = load_returns().iloc[WARMUP:]
    bm = benchmarks(ret)
    out = {}
    for name, w in generate_strategy_variants().items():
        if name in ("A_teknik", "B_sektor", "C_temel", "D_makro", "E_haber"):
            out[name] = (1 + run_backtest(w.reindex(ret.index), ret)["net"].fillna(0)).cumprod()
    out["XU100"] = (1 + bm["xu100"].fillna(0)).cumprod()
    out["equal-weight B&H"] = (1 + bm["esit_agirlik"].fillna(0)).cumprod()
    return pd.DataFrame(out)


# ---------- 2. MCP + harness ----------
def tool_table() -> pd.DataFrame:
    async def run():
        async with Client(build_server()) as c:
            return (await c.list_tools()).tools
    tools = asyncio.run(run())
    used_in = {t: [s for s, ts in STATE_TOOLS.items() if t in ts] for t in TOOL_EN}
    return pd.DataFrame([{"tool": f"`{t.name}`", "inputs": ", ".join(t.input_schema.get("properties", {})),
                          "output": TOOL_EN.get(t.name, ""), "states": ", ".join(used_in.get(t.name, []))}
                         for t in tools])


def state_table() -> pd.DataFrame:
    rows = []
    for i, s in enumerate(STATES):
        rows.append({"#": i, "state": s, "allowed tools": ", ".join(f"`{t}`" for t in STATE_TOOLS[s]) or "–",
                     "LLM agent": AGENT_EN.get(AGENT_AT.get(s, ""), "–"),
                     "mandatory": "yes" if s in MANDATORY else "", "safe-failure path": "yes" if s in SAFE_PATH else "",
                     **{f"in {v}": "✓" if s in VERSION_STATES[v] else "" for v in ("C", "D", "E")}})
    return pd.DataFrame(rows).set_index("#")


def gate_table() -> pd.DataFrame:
    rows = [("HARD_BLOCK", "unknown ticker"), ("HARD_BLOCK", "no price / empty close"),
            ("HARD_BLOCK", f"last price older than {G.MAX_PRICE_STALE_DAYS} days before the analysis day"),
            ("HARD_BLOCK", f"a daily move above ±{G.DAILY_LIMIT * 100:.1f}% in 250 days (broken history; BIST limit ±10%)"),
            ("HARD_BLOCK", "any source date later than the analysis day (future information)"),
            ("HARD_BLOCK", f"latest visible statement older than {G.MAX_FUND_AGE_DAYS} days or after the analysis day"),
            ("HARD_BLOCK", "backtest ends after the analysis day"),
            ("ANALYSIS_UNSAFE", "market-closure window (Feb 2023 earthquake)"),
            ("soft", f"latest statement older than {G.STALE_FUND_DAYS} days; no visible statement"),
            ("soft", "a macro series missing on the analysis day"),
            ("soft", f"fewer than {G.MIN_NEWS_180D} KAP disclosures in 180 days (sparse news)"),
            ("soft", "speech transcript quality warning"),
            ("soft", f"a historical statistic based on fewer than {G.MIN_HIST_EVENTS} events"),
            ("soft", f"a historical hit rate below {G.MIN_HIT:.0%} (not better than a coin flip)"),
            ("soft", f"peer group smaller than {G.MIN_PEERS}"), ("soft", "backtest flagged (fewer than 30 entries)"),
            ("soft", "stock listed as not a BIST 100 member on the analysis day (`config/bist100_membership.csv`; "
                     "a period without a record is 'not verified', no warning)"),
            ("result", f"no warnings → PASS; 1–{G.INVESTIGATE_AT - 1} → PASS_WITH_WARNINGS; "
                       f"≥{G.INVESTIGATE_AT} → INVESTIGATE")]
    return pd.DataFrame(rows, columns=["outcome", "rule"])


# ---------- 3. experiments ----------
def experiment_tables() -> tuple:
    s = pd.read_csv("reports/harness_experiments.csv").set_index("version")
    s = s.rename(columns=METRIC_EN)
    pct = [c for c in s.columns if c not in ("case runs", "look-ahead tool calls", "tool calls per run",
                                             "LLM calls per run", "seconds per run")]
    s[pct] = s[pct] * 100
    raw = pd.DataFrame(map(json.loads, open("reports/experiments/raw.jsonl", encoding="utf-8")))
    r1 = raw[raw["run"] == 1]
    piv = r1.pivot_table(index=["ticker", "asof"], columns="version", values="action", aggfunc="first")
    ref = r1.groupby(["ticker", "asof"])["ref_action"].first()
    label = {(t, d): CASE_EN.get(lab, lab) for t, d, lab in CASES}
    piv.insert(0, "reference (rules)", ref)
    piv.insert(0, "case", [label.get(i, "") for i in piv.index])
    order = [(t, d) for t, d, _ in CASES]
    piv = piv.reindex(order)
    for v in "ABCDE":
        piv[v] = [a + ("" if a == r else " ✗") for a, r in zip(piv[v], piv["reference (rules)"])]
    fab = raw[raw["fabricated"] > 0][["ticker", "asof", "version", "run", "fabricated", "numbers", "unsupported"]]
    fab = fab.assign(unsupported=fab["unsupported"].map(lambda x: ", ".join(map(str, x))[:70]))
    rev = raw[(raw["version"] == "E") & (raw["run"] == 1)][["ticker", "asof", "action", "review"]].copy()
    rev["decision"] = rev["review"].map(lambda r: (r or {}).get("decision"))
    rev["final"] = rev["review"].map(lambda r: (r or {}).get("final_action") or "")
    rev["reason (human)"] = rev["review"].map(lambda r: (r or {}).get("reason", ""))
    return s, piv, fab, rev.drop(columns="review"), raw["model"].dropna().unique()


# ---------- 4. decision log ----------
def log_tables() -> tuple:
    with sqlite3.connect(DB) as con:
        a = pd.read_sql_query("SELECT a.analysis_id, a.created_at, a.ticker, a.asof, a.action, a.gate_status, "
                              "t.llm, r.decision, r.final_action, r.reason FROM analyses a "
                              "LEFT JOIN harness_traces t ON a.analysis_id = t.analysis_id "
                              "LEFT JOIN human_reviews r ON a.analysis_id = r.analysis_id ORDER BY a.created_at", con)
    rep = []
    for aid in a["analysis_id"]:
        r = replay_analysis(aid)
        rep.append({"analysis_id": aid, "same fingerprint": r["same"], "stored action": r["stored_action"],
                    "replayed action": r["new_action"], "stored gate": r["stored_gate"], "replayed gate": r["new_gate"],
                    "changed data files": ", ".join(r["changed_data_files"]) or "none",
                    "procedure changed": r["procedure_changed"]})
    fields = pd.DataFrame([
        ("ticker and sector", "`analyses.ticker`, `analyses.sector`"),
        ("analysis timestamp and market-data period", "`created_at`, `asof`, `data_start`, `data_end`"),
        ("data versions and publication timestamps", "`data_versions` (file hashes), `observed` (event days, CPI month)"),
        ("tools called and tool outputs", "`tools_called`; full calls and results in `harness_traces.trace`"),
        ("technical / sector / fundamental / macro states", "`observed`, `conditions`"),
        ("news and speech source metadata", "`observed.news` (KAP links), `observed.speech` (source, time, stance)"),
        ("full evidence bundle and committed evidence", "`observed`, `history`, `committed`"),
        ("strategy variant and backtest result", "`backtest`"), ("risk / data-quality gate status", "`gate_status`, `gate`"),
        ("agent conclusion and explanation", "`action`, `secondary`, `explanation`"),
        ("human review result and reason", "`human_reviews.decision`, `final_action`, `reason`"),
        ("which model and how it behaved", "`harness_traces.llm`, `summary` (numbers checked, unsupported)")],
        columns=["spec §28 item", "where it is stored"])
    return a.set_index("analysis_id"), pd.DataFrame(rep).set_index("analysis_id"), fields


# ---------- grafikler ----------
def fig_equity(eq: pd.DataFrame) -> str:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    style(ax, "y")
    cols = {"A_teknik": C["s1"], "B_sektor": C["s2"], "C_temel": "#1baf7a", "D_makro": "#eda100", "E_haber": "#e87ba4"}
    for c, col in cols.items():
        ax.plot(eq.index, eq[c], color=col, linewidth=1.6, label=c)
    ax.plot(eq.index, eq["XU100"], color=C["muted"], linewidth=1.6, linestyle="--", label="XU100")
    ax.plot(eq.index, eq["equal-weight B&H"], color=C["text"], linewidth=1.2, linestyle=":", label="equal-weight B&H")
    for c in ("C_temel", "XU100", "equal-weight B&H"):
        ax.text(eq.index[-1], eq[c].iloc[-1], f" {eq[c].iloc[-1]:.2f}×", fontsize=7.5, color=C["text2"], va="center")
    others = [eq[c].iloc[-1] for c in ("A_teknik", "B_sektor", "D_makro", "E_haber")]
    ax.text(eq.index[-1], np.mean(others) * 0.93, f" A,B,D,E: {min(others):.2f}–{max(others):.2f}×", fontsize=7.5,
            color=C["text2"], va="center")
    ax.set_yscale("log")
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator
    ax.yaxis.set_major_locator(FixedLocator([1, 1.5, 2, 3, 4, 5, 6]))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.set_ylabel("growth of 1 TRY after costs (log scale)", fontsize=9, color=C["text2"])
    ax.set_title("Figure 1. No strategy variant beats equal-weight buy-and-hold; extra layers do not add return",
                 loc="left", fontsize=10.5, color=C["text"])
    ax.legend(frameon=False, fontsize=8, labelcolor=C["text2"], ncol=2, loc="upper left")
    return save_fig(fig, "06_equity_curves.png")


def fig_experiments(s: pd.DataFrame) -> str:
    m = ["same as reference", "unsupported conclusion", "numerical hallucination", "evidence completeness"]
    x = np.arange(len(s.index))
    w = 0.2
    fig, ax = plt.subplots(figsize=(8, 3.8))
    style(ax, "y")
    for i, (c, col) in enumerate(zip(m, (C["s1"], C["s2"], "#e34948", "#1baf7a"))):
        ax.bar(x + (i - 1.5) * w, s[c], width=w - 0.02, color=col, label=c)
        for xi, v in zip(x, s[c]):
            if v < 1:
                ax.text(xi + (i - 1.5) * w, 1, "0", ha="center", va="bottom", fontsize=7, color=C["text2"])
    ax.set_xticks(x, [f"{v}" for v in s.index])
    ax.set_ylabel("% of case runs / numbers", fontsize=9, color=C["text2"])
    ax.set_ylim(0, 105)
    ax.set_title("Figure 2. Rules + evidence (D, E) give the reference action with complete, number-safe explanations",
                 loc="left", fontsize=10.5, color=C["text"])
    ax.legend(frameon=False, fontsize=8, labelcolor=C["text2"], ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.1))
    return save_fig(fig, "06_experiments.png")


# ---------- rapor ----------
def scan_tables() -> tuple:
    """Kaydedilmiş tarama tablolarından özet ve 'neden aday değil' tablosu (yeniden hesaplamaz)."""
    from src.harness.scan import from_csv, select
    summ, why = [], []
    for f in SCAN_FILES:
        sc = from_csv(f)
        acts = pd.Series([r["action"] for r in sc["rows"]]).value_counts()
        summ.append({"scan day": sc["asof"], "stocks": len(sc["rows"]),
                     "actions": ", ".join(f"{k} {v}" for k, v in acts.items()),
                     "candidates": ", ".join(sc["candidates"]) or "none",
                     "sent to full harness (max 3)": ", ".join(select(sc)) or "none"})
        for x in sc["also_secondary"]:
            why.append({"scan day": sc["asof"], "stock": x["ticker"], "status": f"secondary ({x['action']})",
                        "why not a candidate": "; ".join(x["why"])})
        for x in sc["near_misses"]:
            why.append({"scan day": sc["asof"], "stock": x["ticker"],
                        "status": f"near miss (lag {x['sector_lag_10'] * 100:+.1f}%)",
                        "why not a candidate": "; ".join(x["why"])})
    return pd.DataFrame(summ), pd.DataFrame(why)


def build() -> str:
    notes = load_notes(NOTES_FILE)
    bt, bench, cost = backtest_tables()
    bsec, broll, bwf, bws = backtest_extra()
    la = lookahead_check()
    eq = equity_curves()
    tools, states, gate = tool_table(), state_table(), gate_table()
    exp, piv, fab, rev, models = experiment_tables()
    log, rep, fields = log_tables()
    f1, f2 = fig_equity(eq), fig_experiments(exp)
    variants = pd.DataFrame([(k, *v) for k, v in VARIANTS.items()], columns=["variant", "name", "rule"])
    versions = pd.DataFrame([(k, v) for k, v in VERSION_EN.items()], columns=["version", "definition"])
    cases = pd.DataFrame([(t, d, CASE_EN.get(lab, lab)) for t, d, lab in CASES], columns=["ticker", "date", "what it tests"])
    scan_sum, scan_why = scan_tables()
    mem = pd.DataFrame([("temporal", "prices, indicators, macro, statements, event histories",
                         "`data/` caches and `reports/*.parquet`; file hashes stored with every analysis"),
                        ("episodic", "past cases: state → evidence → action → backtest → human decision",
                         "SQLite `analyses` + `human_reviews`; agents see up to 3 earlier cases of the same stock, "
                         "only with an earlier analysis date"),
                        ("procedural", "workflow order, action set, gate outputs, rules",
                         f"`PROCEDURE` in `decision_log.py` ({len(PROCEDURE['rules'])} rules), hashed into "
                         "`procedure_version`")], columns=["memory", "content", "implementation"])
    fmt_bt = {"total %": "{:.0f}", "annual %": "{:.0f}", "vol %": "{:.0f}", "max DD %": "{:.0f}",
              "win days %": "{:.0f}", "in market %": "{:.0f}", "costs %": "{:.0f}", "vs equal-weight pt": "{:+.0f}",
              "vs XU100 pt": "{:+.0f}"}

    md = f"""# Report 6 — Backtests, MCP Tools, Stateful Harness, Experiments and Decision Log

_{DISCLAIMER}_

Generated by `python -m src.reports.build_reports --only 6`. Backtest curves, the look-ahead self-test, the MCP
tool list and the replays are recomputed when the report is built (no API key needed).

{stamp(SOURCES)}

This report covers four deliverables of the project brief (§36): **backtesting results and benchmark comparisons**,
**MCP tool definitions and the stateful harness implementation**, the **agent-harness experiment comparison**, and
the **decision-log and replay demonstration**.

{note(notes, "summary")}

## 1. Backtests and benchmarks

### 1.1 Strategy variants (spec §29)

All variants are long-only and rebalanced weekly. Each holds up to {TOP_K} stocks with equal weights, and unused
weight stays in cash at 0%. The decision is made at day t's close and earns day t+1's return. A one-way cost of
{COST_ONE_WAY * 100:.2f}% (commission + slippage) is charged on every change of weight. The first {WARMUP} trading days
are indicator warm-up and are not evaluated.

{md_table(variants, index=False)}

### 1.2 Results

**Table 1. Strategy results after costs (full period and halves; first half = in-sample, second = out-of-sample)**

{md_table(bt, fmt=fmt_bt)}

**Table 2. Benchmarks over the same periods**

{md_table(bench, default="{:.0f}")}

![Figure 1]({f1})

**Table 3. Cost sensitivity (total return %, full period)**

{md_table(cost, default="{:.0f}")}

**Table 3b. Sector-neutral benchmark and trade-level results.** The sector-neutral benchmark holds, every day, the
same sector weights as the strategy but every stock of each sector in equal weight (sector members outside the
30-stock universe included), at no cost. The difference is what the strategy gained or lost by **picking stocks
inside sectors**. A trade is one continuous holding of a stock, entry to exit, after one-way costs on both sides;
trades still open at the end of a period are left out.

{md_table(bsec, fmt={"total %": "{:.0f}", "sector-neutral %": "{:.0f}", "vs sector-neutral pt": "{:+.0f}",
                     "closed trades": "{:,.0f}", "winning trades %": "{:.0f}", "median trade %": "{:+.2f}",
                     "trades beating XU100 %": "{:.0f}", "avg days held": "{:.1f}"})}

**Table 3c. Rolling ~6-month windows (126 trading days, not overlapping)**

{md_table(broll, fmt={"worst window vs XU100 pt": "{:+.0f}", "best window vs XU100 pt": "{:+.0f}"})}

**Table 3d. Walk-forward selection.** At the start of each ~6-month test window the variant with the highest
Sharpe ratio over the previous 252 trading days (after costs, data before the window only) is chosen and held for
the window. The windows are joined into one out-of-sample run with switching costs.

{md_table(bwf, index=False, fmt={"training Sharpe": "{:.2f}", "chosen return %": "{:+.0f}", "XU100 %": "{:+.0f}"})}

**Table 3e. The same out-of-sample period ({bwf["test window"].iloc[0][:10]} – {bwf["test window"].iloc[-1][-10:]})**

{md_table(bws, index=False, fmt={"total %": "{:.0f}", "Sharpe": "{:.2f}", "max DD %": "{:.0f}"})}

**Table 4. Look-ahead self-test of the engine.** A deliberately impossible signal (it holds today's winners)
is run once without the one-day shift and once with it.

{md_table(la, fmt={"total %": "{:,.0f}", "max DD %": "{:.0f}"})}

{note(notes, "backtest")}

## 2. MCP tools and the stateful harness

### 2.1 MCP server (spec §22)

`src/mcp_server/server.py` exposes 15 deterministic tools (mcp 2.3 `MCPServer`). Each tool validates its
inputs. Unknown tickers, unknown strategies and dates after the last data day raise a `ToolError` whose message
reaches the agent. All dates are point-in-time (see the data dictionary).

{md_table(tools, index=False)}

### 2.2 State machine (spec §23) and agents (spec §20)

The order of states is fixed in code. The LLM cannot choose the next state, and each state opens only the tools
listed. A request for any other tool is blocked and counted as an invalid call. After a hard block, the analysis
states are skipped, but the quality check, risk gate, explanation, human review and logging are not (safe-failure
path). The last three columns show which states each experiment version runs (Section 3).

{md_table(states)}

Deterministic workers (data loading, indicators, events, backtests) are tools, not agents. Six LLM agents only
interpret tool outputs:

- **Market**, **Fundamental & Macro**, **News & Speech**, **Strategy** and **Risk** each summarise their state's
  tool results.
- The **Orchestrator** writes the final explanation from the committed evidence.

The action itself is chosen by rules (`evidence.decide`) from the 8 allowed labels: {", ".join(f"`{x}`" for x in ACTIONS)}.

### 2.3 Risk and data-quality gate (spec §26)

{md_table(gate, index=False)}

### 2.4 Memory (spec §24)

{md_table(mem, index=False)}

{note(notes, "harness")}

### 2.5 Universe scan mode (spec §33)

The spec's end-to-end example asks a question about the whole universe: *"Find BIST 100 stocks that are lagging a
strong sector but may be entering a historically favourable catch-up state."* `src/harness/scan.py` runs the rule
layer for all 30 stocks on the scan day (no LLM), lists the stocks whose action is the asked label, explains why
the nearest others are not candidates, and sends at most three candidates through the full harness (LLM
explanation, human review, log). For the catch-up question the three with the largest sector lag go first; this
order is for display and was not tested as a forecast.

**Table 4b. Spec §33 steps and where the system does them**

{md_table(pd.DataFrame(SCAN_STEPS, columns=["spec step", "in the system"]), index=False)}

**Table 4c. Two committed scans** (`python -m src.harness.scan <day>`)

{md_table(scan_sum, index=False)}

**Table 4d. Why the other lagging stocks are not candidates**

{md_table(scan_why, index=False)}

{note(notes, "scan")}

## 3. Harness experiments A–E (spec §30)

**Table 5. Versions**

{md_table(versions, index=False)}

**Table 6. Cases** (each run twice; run 2 uses a separate LLM cache so it is a real repetition)

{md_table(cases, index=False)}

Model: {", ".join(f"`{m}`" for m in models)} (temperature 0) for every row. The reference action is the rule-based
result of the evidence bundle.

**Table 7. Results by version (percentages unless stated)**

{md_table(exp, fmt={"case runs": "{:.0f}", "look-ahead tool calls": "{:.0f}", "tool calls per run": "{:.1f}",
                    "LLM calls per run": "{:.1f}", "seconds per run": "{:.0f}"}, default="{:.1f}")}

![Figure 2]({f2})

**Table 8. Action per case and version (run 1; ✗ = differs from the reference)**

{md_table(piv)}

**Table 9. Runs with fabricated numbers after the validator fix** (numbers in the explanation that appear in no tool
output). Before the fix, six B runs were flagged; every flagged number was a moving-average window ("20, 50 and
200-day"), see Section 5.

{md_table(fab, index=False) if len(fab) else "None."}

**Table 10. Human reviews in version E (run 1; run 2 reuses the same decision)**

{md_table(rev, index=False)}

{note(notes, "experiments")}

## 4. Decision log and replay (spec §27–28)

**Table 11. Where each required log item is stored**

{md_table(fields, index=False)}

**Table 12. Real decision log** (`reports/decision_log.sqlite`; experiment runs are kept in a separate
`reports/experiments/experiment_log.sqlite`, demos in `decision_log_demo.sqlite`)

{md_table(log)}

**Table 13. Replay of every real record with today's code and data**

{md_table(rep)}

{note(notes, "replay")}

## 5. Problems found by the system's own checks

{note(notes, "findings")}

## 6. Limitations

{note(notes, "limitations")}

## 7. Conclusion

{note(notes, "conclusion")}
"""
    return str(write("06_system_harness.md", md))
