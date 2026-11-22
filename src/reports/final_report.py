"""Final teknik rapor (PDF Bölüm 36, 38). Kendi başına okunur; ayrıntılar için Rapor 0-6'ya atıf yapar.

Anahtar sayılar rapor modüllerinin aynı fonksiyonlarından hesaplanır (Rapor 1-6 ile tutarlı kalır); örnek analizler
bugünkü kodla yeniden üretilir. Yorum metni reports/research/notes/final_report.md dosyasındadır.

  python -m src.reports.build_reports --only 7
  python -m src.reports.pdf --final
"""
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from src.reports.common import C, DISCLAIMER, load_notes, md_table, note, save_fig, write

NOTES_FILE = "final_report.md"
CASE_WHY = {
    ("ASELS", "2026-11-06"): "top-third earnings, no shock or negative news",
    ("TURSG", "2026-11-06"): "shock day + two negative KAP disclosures; profit turned into a loss",
    ("PETKM", "2026-11-06"): "bullish and bearish technical events in the same 3 days",
    ("KRDMD", "2026-11-06"): "earnings candidate, but a −10% shock day 1 day earlier",
    ("ASELS", "2023-02-15"): "earthquake market-closure window",
    ("YOKBOYLE", "2026-11-06"): "ticker not in the universe",
}


# ---------- mimari diyagramı ----------
def fig_architecture() -> str:
    fig, ax = plt.subplots(figsize=(8.2, 8.6))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")

    def box(x, y, w, h, title, body, face, edge):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.4,rounding_size=1.5", fc=face, ec=edge, lw=1.2))
        ax.text(x + w / 2, y + h - 2.2, title, ha="center", va="top", fontsize=8.6, weight="bold", color=C["text"])
        ax.text(x + w / 2, y + h - 6.0, body, ha="center", va="top", fontsize=7.1, color=C["text2"], linespacing=1.35)

    def arrow(x1, y1, x2, y2, text=None, dashed=False):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=10, lw=1.1,
                                     color=C["text2"], linestyle="--" if dashed else "-"))
        if text:
            ax.text((x1 + x2) / 2 + 1, (y1 + y2) / 2, text, fontsize=6.6, color=C["text2"], va="center")

    blue, blue_e = "#eaf2fc", C["s1"]
    grey, grey_e = "#f3f2ef", C["muted"]
    orange, orange_e = "#fdeee7", C["s2"]
    green, green_e = "#e6f6ef", "#1baf7a"
    box(2, 86, 96, 12, "Data sources (cached, versioned by file hash)",
        "İş Yatırım prices · Fintables statements · KAP publication times & disclosures · TCMB EVDS / TÜİK CPI · Fed table\n"
        "RSS headlines · Fed & TCMB official speeches (Whisper / official text)", grey, grey_e)
    box(2, 66, 96, 15, "Deterministic workers — point-in-time (no LLM)",
        "features: SMA/EMA/RSI/MACD/Bollinger/ATR/KAMA/Supertrend/Ichimoku, support/resistance, sector lag\n"
        "event tables: reversals, sector catch-up, calendar, earnings (KAP event day), news signals, speech claims\n"
        "backtest engine: 1-day signal shift, 0.15% cost, XU100 & equal-weight benchmarks", blue, blue_e)
    box(2, 52, 96, 10, "MCP server — 15 tools (mcp 2.3)",
        "market · indicators · events · support/resistance · sector rank · weekday · fundamentals · macro · news · speech\n"
        "backtest · data-quality gate · evidence bundle — inputs validated, future dates rejected", blue, blue_e)
    box(2, 24, 60, 24, "Stateful harness — 17 fixed states",
        "START → … → LOAD_MARKET_DATA → CHECK_DATA_QUALITY →\nTECHNICALS → SECTOR → PATTERNS → FUNDAMENTALS →\n"
        "MACRO → NEWS & SPEECH → BUILD_EVIDENCE →\nPREVIEW_STRATEGIES → RUN_BACKTEST → RISK_GATE →\n"
        "GENERATE_EXPLANATION → HUMAN_REVIEW → LOG_AND_CLOSE\n\nper-state tool permissions · safe-failure path\n"
        "on HARD_BLOCK / ANALYSIS_UNSAFE", orange, orange_e)
    box(66, 24, 32, 24, "LLM agents (Gemini)",
        "Market · Fundamental & Macro\nNews & Speech · Strategy\nRisk · Orchestrator\n\n"
        "interpret tool output only;\nevery number checked;\none correction retry", orange, orange_e)
    box(2, 4, 44, 15, "Evidence + rules + gate",
        "observed context (100+ fields)\ncommitted evidence E1…En (code-written)\n"
        "rule action from 8 labels · gate:\nPASS … INVESTIGATE / UNSAFE / HARD_BLOCK", green, green_e)
    box(52, 4, 46, 15, "Human review → decision log",
        "ACCEPT / MODIFY / REJECT + mandatory reason\nSQLite: analyses, reviews, traces\n"
        "replay: data hash + procedure hash\nepisodic memory (earlier cases only)", green, green_e)
    arrow(50, 86, 50, 81.4)
    arrow(50, 66, 50, 62.4)
    arrow(32, 52, 32, 48.4, "MCP calls")
    arrow(62.4, 36, 65.6, 36)
    arrow(24, 24, 24, 19.4)
    arrow(46.4, 11.5, 51.6, 11.5)
    arrow(75, 19.4, 82, 23.6, "memory", dashed=True)
    ax.text(50, 0.5, "No broker connection: the output is an analytical state for a human, never an order.",
            ha="center", fontsize=7.2, color=C["text2"], style="italic")
    return save_fig(fig, "final_architecture.png")


# ---------- anahtar tablolar ----------
def research_summary() -> pd.DataFrame:
    from src.reports import r1_technical as R1, r2_sector as R2, r3_weekday as R3, r4_earnings as R4
    e1 = R1.load_events()
    e1 = e1[~e1["quake_window"]]
    m1 = R1.main_table(e1, R1.baseline().iloc[0])
    ot1, _ = R1.oos_table(e1)
    s2 = R2.load_events()
    lad = R2.filter_ladder(s2, R2.baseline().loc["gap"])
    oos2 = R2.oos_table(s2)
    _, port, dow, valid = R3.portfolio()
    w = pd.read_csv("reports/weekday_portfolio.csv").set_index("day")
    bt3, _ = R3.wed_thu_backtest()
    sp4 = pd.read_csv("reports/earnings_spread_tests.csv").set_index("pencere")
    e4 = R4.load_events()
    fam4, met, _ = R4.test_family(e4, sp4)          # Rapor 4 ile aynı 22 testlik BH ailesi
    met = met.set_index(["metric", "window"])
    q4 = fam4[fam4["family"] == "net income spread vs XU100"].set_index("test")["q (all tests)"]
    mt4 = R4.method_table(e4)
    a_d0, b_d0 = mt4.iloc[0]["event day pt"], mt4.iloc[1]["event day pt"]
    bt = pd.read_csv("reports/backtest_strategies.csv").set_index(["strateji", "donem"])["toplam_getiri"] * 100
    rows = [
        ("1 Technical reversals", f"excess vs XU100 of 12 event types: {m1['excess 5d %'].min():+.2f} to "
         f"{m1['excess 5d %'].max():+.2f}% in 5 days; raw gains ≈ any-day drift (+{R1.baseline().iloc[0]['mean_5']:.2f}%)",
         f"no type with q < 0.05; {int(ot1['same sign'].sum())}/12 keep sign out of sample", "context only"),
        ("2 Sector catch-up", f"harness rule (no shock, strong sector, recovering, not weak earnings): "
         f"{lad.iloc[-1]['gap 10d']:+.2f} pt vs peers in 10 days (baseline {lad.iloc[0]['gap 10d']:+.2f})",
         f"only q < 0.05 result; {oos2.loc['harness catch-up', 'gap 10d first']:+.2f} / "
         f"{oos2.loc['harness catch-up', 'gap 10d second']:+.2f} in the two halves; ~half still lag after 20 days",
         "`POTENTIAL_CATCH_UP_CANDIDATE`"),
        ("3 Weekday & multi-day", f"Wednesday {w.loc['Çar', 'diff']:+.2f}, Thursday {w.loc['Per', 'diff']:+.2f} pt vs "
         f"other days; big down days continue (−1.35 pt in 5 days)",
         f"Wed-buy/Thu-sell: {bt3.iloc[0]['total return %']:+.0f}% before costs, {bt3.iloc[1]['total return %']:+.0f}% at 0.15% "
         f"(XU100 +{bt3.iloc[0]['XU100 %']:.0f}%)", "not used for actions"),
        ("4 Earnings reaction", f"top − bottom net-income third: {sp4.loc['pre5', 'ort_fark']:+.2f} pt before "
         f"(q = {q4['5 days before']:.2f}), {sp4.loc['d0', 'ort_fark']:+.2f} pt on the event day "
         f"(q = {q4['event day']:.3f}); operating income "
         f"{met.loc[('operating income YoY (non-financial)', 'event day'), 'top − bottom (pt)']:+.2f}, revenue "
         f"{met.loc[('revenue YoY (non-financial)', 'event day'), 'top − bottom (pt)']:+.2f}; pre-registered H2 "
         f"EBITDA-margin change {met.loc[('H2 EBITDA margin change, pt', 'event day'), 'top − bottom (pt)']:+.2f} "
         f"(q = {met.loc[('H2 EBITDA margin change, pt', 'event day'), 'q (all tests)']:.3f}), H1/H3/H4 not confirmed",
         f"event day {a_d0:+.2f} pt with previous-quarter cut-offs, {b_d0:+.2f} pt ranking only companies already "
         f"reported; no drift after day 4 (net-margin change "
         f"{met.loc[('net margin change, pt (non-financial)', 'days 1–20'), 'top − bottom (pt)']:+.2f} pt over days 1–20 "
         f"is borderline, q = {met.loc[('net margin change, pt (non-financial)', 'days 1–20'), 'q (all tests)']:.3f})",
         "`FUNDAMENTAL_IMPROVEMENT_CANDIDATE`"),
        ("5 Macro, news, speech", f"XU100 ×7.5 in TRY but ×1.4 in USD; macro risk flag lowered returns "
         f"(C {bt[('C_temel', 'tum')]:.0f}% → D {bt[('D_makro', 'tum')]:.0f}%); "
         "only 'new contract' KAP news moves prices (+0.5 pt)", "3 of 33 speech claims verified and admitted as evidence",
         "context, gate warnings, blockers"),
    ]
    return pd.DataFrame(rows, columns=["research scenario (report)", "key result", "robustness", "harness use"])


def backtest_summary() -> pd.DataFrame:
    b = pd.read_csv("reports/backtest_strategies.csv")
    b = b[b["strateji"].isin(["A_teknik", "B_sektor", "C_temel", "D_makro", "E_haber"])]
    piv = b.pivot_table(index="strateji", columns="donem", values="toplam_getiri") * 100
    sh = b[b["donem"] == "tum"].set_index("strateji")
    out = pd.DataFrame({"full %": piv["tum"], "first half %": piv["ilk_yari"], "second half %": piv["ikinci_yari"],
                        "Sharpe (full)": sh["sharpe"], "max DD %": sh["max_dusus"] * 100,
                        "entries": sh["islem_sayisi"]})
    xu = b[b["strateji"] == "A_teknik"].set_index("donem")
    xu = (xu["toplam_getiri"] - xu["xu100_fark"]) * 100
    ew = b[b["strateji"] == "A_teknik"].set_index("donem")["kiyas_getiri"] * 100
    out.loc["XU100"] = [xu["tum"], xu["ilk_yari"], xu["ikinci_yari"], np.nan, np.nan, np.nan]
    out.loc["equal-weight buy & hold"] = [ew["tum"], ew["ilk_yari"], ew["ikinci_yari"], np.nan, np.nan, np.nan]
    return out.rename_axis("strategy")


def experiment_summary() -> pd.DataFrame:
    s = pd.read_csv("reports/harness_experiments.csv").set_index("version")
    out = pd.DataFrame({"same action as evidence-based reference %": s["referansla_ayni"] * 100,
                        "unsupported conclusion %": s["desteksiz_sonuc"] * 100,
                        "fabricated numbers %": s["sayisal_halusinasyon"] * 100,
                        "evidence completeness %": s["kanit_tamligi"] * 100,
                        "reproducible %": s["tekrarlanabilirlik"] * 100,
                        "tool calls / run": s["arac_cagrisi"], "LLM calls / run": s["llm_cagrisi"]})
    return out


def case_table() -> tuple:
    from src.harness.evidence import run_analysis
    raw = pd.DataFrame(map(json.loads, open("reports/experiments/raw.jsonl", encoding="utf-8")))
    rev = raw[(raw["version"] == "E") & (raw["run"] == 1)].set_index(["ticker", "asof"])["review"]
    rows, asels = [], None
    for (t, d), why in CASE_WHY.items():
        b = run_analysis(t, d)
        r = rev.get((t, d)) or {}
        rows.append({"case": f"{t} {d}", "why": why, "gate": b["gate"]["status"], "action (rules)": f"`{b['action']}`",
                     "evidence items": len(b["committed"]),
                     "human (Step 14)": r.get("decision", "") + (f" → {r['final_action']}" if r.get("final_action") else "")})
        if t == "ASELS" and d == "2026-11-06":
            asels = b
    ev = pd.DataFrame([{"id": e["id"], "tool": f"`{e['tool']}`", "committed evidence (system output, Turkish)": e["text"]}
                       for e in asels["committed"]])
    return pd.DataFrame(rows), ev


def deliverables() -> pd.DataFrame:
    rows = [
        ("Python source code and environment instructions", "`src/`, `tests/`, `README.md`, `requirements.txt`, "
         "`.env.example`; data archive via `python -m src.data.pack_data`"),
        ("Fixed BIST 100 study universe and sector map", "`config/universe.csv`; Report 0 §1; Report 2 §2"),
        ("Data dictionary (structured and unstructured features)", "Report 0 (`00_data_dictionary.md`)"),
        ("Technical reversal-event report", "Report 1"), ("Sector laggard / catch-up research report", "Report 2"),
        ("Weekday and multi-day pattern research report", "Report 3"),
        ("Quarterly fundamentals and post-disclosure price-reaction report", "Report 4"),
        ("Macro and contextual evidence integration report", "Report 5"),
        ("MCP tool definitions and stateful harness implementation", "`src/mcp_server/`, `src/harness/`; Report 6 §2"),
        ("Decision-log and replay demonstration", "`reports/decision_log.sqlite`; Report 6 §4; "
         "`python -m src.harness.decision_log`"),
        ("Backtesting results and benchmark comparisons", "Report 6 §1; `reports/backtest_*.csv`"),
        ("Agent-harness experiment comparison", "Report 6 §3; `reports/harness_experiments.csv`, `reports/experiments/`"),
        ("Final technical report and classroom demonstration", "this document; demo script in the presentation"),
    ]
    return pd.DataFrame(rows, columns=["deliverable (brief §36)", "where"])


def build() -> str:
    notes = load_notes(NOTES_FILE)
    arch = fig_architecture()
    rs, bt, ex = research_summary(), backtest_summary(), experiment_summary()
    cases, asels_ev = case_table()
    cover = notes.get("cover", "")

    md = f"""{cover}

# Final Technical Report — An Evidence-First Agentic Harness for BIST 100 Analytics

_{DISCLAIMER}_

{note(notes, "abstract")}

## 1. Problem and purpose

{note(notes, "problem")}

## 2. Data and point-in-time design

{note(notes, "data")}

## 3. System architecture

![Figure 1. Architecture]({arch})

{note(notes, "architecture")}

## 4. Research findings

**Table 1. Five research scenarios in one table** (details in Reports 1–5)

{md_table(rs, index=False)}

{note(notes, "research")}

## 5. Strategies and backtests

**Table 2. Strategy variants A–E after 0.15% one-way costs** (Report 6 §1; first half 2022-10 – 2024-11,
second half 2024-11 – 2026-11)

{md_table(bt, fmt={"full %": "{:.0f}", "first half %": "{:.0f}", "second half %": "{:.0f}", "max DD %": "{:.0f}",
                   "entries": "{:,.0f}"})}

{note(notes, "backtest")}

## 6. Harness experiments A–E

**Table 3. What each layer adds** (6 cases × 2 runs, Gemini 3.1 Flash-Lite, temperature 0; Report 6 §3)

{md_table(ex, default="{:.1f}")}

{note(notes, "experiments")}

## 7. Worked examples: from evidence to an analytical state

**Table 4. Six cases, recomputed with the final code**

{md_table(cases, index=False)}

**Table 5. Committed evidence for ASELS on 2026-11-06** (the only sentences the explanation may use)

{md_table(asels_ev, index=False)}

{note(notes, "examples")}

## 8. Decision log, replay and human review

{note(notes, "log")}

## 9. Seven distinctions the project demonstrates

{note(notes, "distinctions")}

## 10. Problems found and fixed

{note(notes, "findings")}

## 11. Limitations and threats to validity

{note(notes, "limitations")}

## 12. Conclusion and future work

{note(notes, "conclusion")}

## Appendix A. Deliverables

{md_table(deliverables(), index=False)}

## Appendix B. Reproducing this report

{note(notes, "reproduce")}
"""
    return str(write("final_report.md", md))
