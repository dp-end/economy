"""Veri sözlüğü (PDF Bölüm 36): her yapısal ve yapısal olmayan özellik.

Açıklamalar bu dosyada elle yazılır; tip, doluluk ve aralık gerçek dosyalardan okunur. tests/test_reports.py,
herhangi bir veri setinde açıklaması olmayan (ya da artık var olmayan) sütun bulursa başarısız olur, böylece
sözlük koddan kopamaz.

  python -m src.reports.build_reports --only 0
"""
import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from src.reports.common import DISCLAIMER, md_table, write

PRICE_COLS = ["High", "Low", "Close", "Volume", "Turnover_TL"]
EVENT_RET = {
    **{f"fwd_{h}": f"stock return over the next {h} trading days (close to close) [fraction]" for h in (1, 3, 5, 10, 20)},
    **{f"exc_{h}": f"`fwd_{h}` minus XU100 return over the same days [fraction]" for h in (1, 3, 5, 10, 20)},
    **{f"gap_{h}": f"stock's next-{h}-day return minus its peers' median next-{h}-day return [fraction]"
       for h in (5, 10, 20)},
}
WIN = {"pre5": "abnormal return vs XU100, days −5…−1 before the event day [fraction]",
       "d0": "abnormal return vs XU100 on the event day [fraction]",
       "d0_1": "abnormal return vs XU100, days 0…1 [fraction]",
       "d0_4": "abnormal return vs XU100, days 0…4 [fraction]",
       "post1_5": "abnormal return vs XU100, days 1…5 after the event day [fraction]",
       "post1_20": "abnormal return vs XU100, days 1…20 [fraction]"}
QUAKE = "True if the window touches the Feb 2023 earthquake closure; such rows are excluded from statistics"
EVENT_DAY = "first trading day on which the market could react (after publication; post-18:00 filings → next day)"
KAP_DATES = {
    "ticker": "stock code", "period_end": "quarter end of the financial statement",
    "window_start": "first day of the legal filing window (legal rule)",
    "deadline": "legal filing deadline (legal rule; used only as a check)",
    "publish_time": "real KAP publication timestamp (Europe/Istanbul) — the point-in-time anchor",
    "kap_index": "KAP disclosure id (link: kap.org.tr/tr/Bildirim/<id>)",
    "event_day": EVENT_DAY, "lag_days": "calendar days from quarter end to publication",
    "after_deadline": "True if published after the legal deadline (15% of filings)"}
SPEECH_META = {
    "source_id": "speech id (`fed_YYYYMMDD`, `tcmb_YYYYMMDD`)",
    "start": "segment start in the recording [seconds]; NaN for official text",
    "end": "segment end [seconds]; NaN for official text",
    "text": "**unstructured**: transcript sentence (Whisper) or sentence of the official text",
    "avg_logprob": "Whisper mean log-probability of the segment (transcription confidence); NaN for text",
    "segment_no": "order of the segment in the speech", "kind": "`video` (transcribed audio) or `text` (official text)",
    "language": "`en` / `tr`", "speaker": "speaker and role", "channel": "official channel (Federal Reserve, TCMB)",
    "title": "event title", "media_url": "URL of the audio/video file", "page_url": "official event page",
    "official_text_url": "official transcript/text used as reference",
    "publish_time": "publication time of the speech (Europe/Istanbul)",
    "event_day": "first BIST trading day on which the speech could matter (NaT if not set)"}
CLAIM = {"values": "numbers (%) extracted from the sentence by rules",
         "variable": "data variable the claim refers to (`fed_faiz_araligi`, `tufe_yillik`, `tcmb_politika_faizi`) or empty",
         "status": "VERIFIED / CONSISTENT_WITH_HISTORY / FORECAST_OR_TARGET / NO_TOOL / CONTRADICTED",
         "tool_value": "value returned by the deterministic data tool as of the speech time"}


def _universe():
    return pd.read_csv("config/universe.csv")


def _prices():
    from src.data.prices import get_stock_history
    return get_stock_history("ASELS").reset_index()


def _fund():
    from src.data.fundamentals import load_all
    return load_all()["TUPRS"].reset_index()


def _features():
    from src.indicators.features import full_features
    return full_features("ASELS").drop(columns=PRICE_COLS).reset_index()


def _pq(p):
    return lambda: pd.read_parquet(p)


def _csv(p):
    return lambda: pd.read_csv(p)


def _sql(table):
    def f():
        with sqlite3.connect("reports/decision_log.sqlite") as con:
            return pd.read_sql_query(f"SELECT * FROM {table}", con)
    return f


def _scan_schema():
    """`scans` tablosunun sütunları (gerçek günlükte tablo ilk taramada oluşur; şema bellekte okunur)."""
    from src.harness.scan import SCAN_SCHEMA
    with sqlite3.connect(":memory:") as con:
        con.executescript(SCAN_SCHEMA)
        return pd.read_sql_query("SELECT * FROM scans", con)


def _raw():
    return pd.DataFrame(map(json.loads, open("reports/experiments/raw.jsonl", encoding="utf-8")))


def flatten_bundle(b: dict) -> dict:
    """Kanıt paketini 'bölüm.alan' anahtarlarına açar (history altındaki olay anahtarları kalıba çevrilir)."""
    out = {}
    for sec in ("meta", "conditions", "gate", "backtest"):
        for k, v in (b.get(sec) or {}).items():
            out[f"{sec}.{k}"] = v
    for sec, d in b["observed"].items():
        for k, v in d.items():
            out[f"observed.{sec}.{k}"] = v
    for k, v in b["history"].items():
        out["history.tech:<event>" if k.startswith("tech:") else f"history.{k}"] = v
    for k, v in b["source_dates"].items():
        out["source_dates.<source>"] = v
    for k in ("action", "secondary", "warning"):
        out[k] = b[k]
    out["committed[].id"], out["committed[].text"] = "E1", b["committed"][0]["text"] if b["committed"] else ""
    out["committed[].value"], out["committed[].tool"] = None, None
    return out


def _bundle():
    from src.harness.evidence import run_analysis
    flat = {}
    for t, d in (("ASELS", "2026-11-06"), ("PETKM", "2026-11-06")):      # PETKM: teknik olay ve geçmiş anahtarları
        flat.update({k: v for k, v in flatten_bundle(run_analysis(t, d)).items() if k not in flat})
    return pd.DataFrame([{k: (json.dumps(v, default=str, ensure_ascii=False)[:60] if isinstance(v, (list, dict)) else v)
                          for k, v in flat.items()}])


DATASETS = [
    dict(key="universe", title="Study universe and sector map", file="config/universe.csv", load=_universe,
         kind="structured", step="1", grain="one row per stock",
         pit="fixed for the whole study (2026 BIST 100 list); survivorship caveat",
         cols={"ticker": "BIST stock code", "yahoo": "Yahoo Finance symbol (validation source only)",
               "sector_pdf": "sector as listed in the project brief",
               "peer_group": "peer group used for sector-relative features (≥3 members to be used)",
               "is_core": "1 = one of the 30 study stocks, 0 = extra peer used only for sector medians"}),
    dict(key="prices", title="Daily prices", file="data/isyatirim/<TICKER>.parquet (example: ASELS)", load=_prices,
         kind="structured", step="2", grain="one row per stock and trading day, 2021-11-01 – 2026-11-06",
         pit="known at that day's close; corporate-action adjusted by İş Yatırım (e.g. CCOLA 1:11 bonus in Aug 2024 "
             "shows no jump); validated against Yahoo (Step 2)",
         cols={"date": "trading day", "High": "daily high, never below the close [TRY]",
               "Low": "daily low, never above the close [TRY]", "Close": "adjusted close (İş Yatırım `HGDG_KAPANIS`) [TRY]",
               "Volume": "traded quantity = TRY turnover / average price [shares]",
               "Turnover_TL": "traded value [TRY]"}),
    dict(key="market_cap", title="Daily market capitalisation", file="data/isyatirim_mcap/<TICKER>.parquet "
         "(example: ASELS)", load=lambda: pd.read_parquet("data/isyatirim_mcap/ASELS.parquet").reset_index(),
         kind="structured", step="1d (review of 22 Nov)", grain="one row per stock and trading day, 2021-11-01 – 2026-11-06",
         pit="known at that day's close. Downloaded on 22 Nov 2026 from the same İş Yatırım endpoint as the prices; "
             "`PD` = unadjusted close × paid-in capital. On three days İş Yatırım updated the capital one day before "
             "the price split (TURSG 2025-06-03, 2026-06-12; BIMAS 2026-05-13); `market_cap()` aligns them",
         cols={"date": "trading day", "PD": "market capitalisation as published by İş Yatırım [TRY]",
               "HG_KAPANIS": "unadjusted close [TRY]", "SERMAYE": "paid-in capital = number of shares (1 TRY nominal)",
               "HAO_PD": "market capitalisation of the free float [TRY]",
               "HGDG_KAPANIS": "adjusted close, same series as `prices.Close` [TRY]"}),
    dict(key="valuation", title="Daily valuation ratios", file="reports/valuation_daily.parquet",
         load=_pq("reports/valuation_daily.parquet"), kind="structured (derived)", step="1d (review of 22 Nov)",
         grain="one row per stock and trading day from the first KAP statement (2022Q1) on",
         pit="market cap at that close; statement values of the latest report public that day; inflation-accounting "
             "figures moved to the money of the latest CPI published that day; net debt not moved (a TRY debt does not "
             "grow with inflation)",
         cols={"date": "trading day", "ticker": "stock code", "market_cap": "aligned market capitalisation [TRY]",
               "report_period": "quarter of the latest public statement", "report_event_day": "its KAP event day",
               "cpi_month": "latest CPI month published by that day (empty for nominal figures)",
               "factor_flow": "CPI factor applied to net income and EBITDA (1 for nominal figures)",
               "factor_stock": "CPI factor applied to equity and minority interests",
               "pe": "P/E = market cap / last-12-month net income in that day's money; empty if net income ≤ 0",
               "pb": "P/B = market cap / parent equity in that day's money; empty if equity ≤ 0",
               "ev": "enterprise value = market cap + net debt + minority interests [TRY]; industrial companies only",
               "ev_ebitda": "EV / last-12-month EBITDA in that day's money; empty if EBITDA ≤ 0",
               "pe_reported": "P/E with net income in the money of its report (as on data screens), for comparison",
               "val_dropped": "ratios left empty and why: `anlamsiz` denominator ≤ 0, `rapor_degeri_yok` statement "
                              "value not known on its event day, `sanayi_disi_ya_da_rapor_degeri_yok` no EBITDA/net debt"}),
    dict(key="fundamentals", title="Quarterly fundamentals (Fintables)", file="data/fintables/*.xlsx → load_all() "
         "(example: TUPRS)", load=_fund, kind="structured", step="2, 7",
         grain="one row per stock and quarter, 2022Q1 – 2026Q3 (2022Q4 missing for 24 of 30 stocks)",
         pit="RAW, NOT point-in-time. Fintables keeps the latest version of every figure. For the 24 companies that "
             "apply inflation accounting (TMS 29) it re-states 2022Q4 – 2023Q3 and re-expresses everything from "
             "2022Q4 on in the purchasing power of the company's latest balance-sheet month (verified against 20 "
             "published figures). Analyses use the point-in-time view in `earnings_events` "
             "(src/data/pit_fundamentals.py), never these raw values",
         cols={"period_end": "quarter end date", "net_income": "net income attributable to parent, quarterly [TRY]",
               "revenue": "net sales, quarterly [TRY]; missing for financials",
               "gross_profit": "gross profit, quarterly [TRY]; missing for financials",
               "op_income": "operating profit, quarterly [TRY]; missing for financials",
               "equity": "shareholders' equity at quarter end [TRY]", "total_assets": "total assets at quarter end [TRY]",
               "ni_yoy": "(net income − same quarter last year) / |same quarter last year| [fraction]",
               "ni_qoq": "(net income − previous quarter) / |previous quarter| [fraction]",
               "roe_q": "quarterly net income / equity [fraction]", "leverage": "total assets / equity [ratio]",
               "net_margin": "net income / revenue [fraction]; non-financials only", "loss": "True if net income < 0"}),
    dict(key="features", title="Technical, support/resistance and relative features", file="computed: "
         "full_features(ticker) (example: ASELS; price columns as above)", load=_features, kind="structured",
         step="3, 5", grain="one row per stock and trading day",
         pit="every value uses only that day's and earlier closes; Ichimoku has no forward-shifted Chikou line; "
             "pivots are used only after they are confirmed (k = 5 days later)",
         cols={"date": "trading day", **{f"sma_{n}": f"{n}-day simple moving average of close [TRY]" for n in (20, 50, 200)},
               **{f"ema_{n}": f"{n}-day exponential moving average [TRY]" for n in (20, 50)},
               "rsi_14": "RSI, 14 days, Wilder smoothing [0–100]", "macd": "EMA12 − EMA26 [TRY]",
               "macd_signal": "9-day EMA of MACD [TRY]", "macd_hist": "MACD − signal [TRY]",
               "bb_mid": "Bollinger middle band = SMA20 [TRY]", "bb_upper": "SMA20 + 2σ [TRY]", "bb_lower": "SMA20 − 2σ [TRY]",
               "bb_bandwidth": "(upper − lower) / middle [fraction]",
               "bb_pctb": "%B = (close − lower) / (upper − lower); <0 below the lower band, >1 above the upper",
               "atr_14": "Average True Range, 14 days, Wilder [TRY]",
               "kama": "Kaufman adaptive moving average (10, 2, 30) [TRY]", "kama_slope": "daily % change of KAMA [fraction]",
               "supertrend": "Supertrend line (ATR 10, multiplier 3) [TRY]", "st_dir": "+1 Supertrend up, −1 down",
               "st_flip": "True on the day the Supertrend direction changes",
               "tenkan": "Ichimoku conversion line (9) [TRY]", "kijun": "Ichimoku base line (26) [TRY]",
               "senkou_a": "Ichimoku span A, plotted 26 days ahead → known today [TRY]",
               "senkou_b": "Ichimoku span B (52), shifted 26 days [TRY]", "cloud_top": "max(span A, span B) [TRY]",
               "cloud_bottom": "min(span A, span B) [TRY]", "cloud_state": "+1 above the cloud, 0 inside, −1 below",
               "vol_z_20": "volume z-score vs the last 20 days", "ret_1d": "1-day return [fraction]",
               "ret_5d": "5-day return [fraction]", "vol_20d": "20-day return volatility, annualised [fraction]",
               "support": "nearest confirmed pivot cluster below the close (250-day lookback, 1.5% tolerance) [TRY]",
               "support_touches": "number of pivots in that support cluster (level strength)",
               "support_age": "trading days since the latest pivot in the support cluster",
               "resistance": "nearest confirmed pivot cluster above the close [TRY]",
               "resistance_touches": "number of pivots in that resistance cluster",
               "resistance_age": "trading days since the latest pivot in the resistance cluster",
               "dist_support": "close / support − 1 (positive = above support) [fraction]",
               "dist_resistance": "resistance / close − 1 (room to resistance) [fraction]",
               **{f"rel_xu_{n}": f"stock {n}-day return − XU100 {n}-day return [fraction]" for n in (5, 10, 20)},
               **{f"sector_lag_{n}": f"stock {n}-day return − median {n}-day return of its peers (itself excluded); "
                  f"negative = behind its sector [fraction]" for n in (5, 10, 20)}}),
    dict(key="macro", title="Macro panel", file="reports/macro_panel.csv", load=_csv("reports/macro_panel.csv"),
         kind="structured", step="8", grain="one row per BIST trading day",
         pit="each value is the last one PUBLISHED by that day (CPI on the 3rd of the next month; Fed on the first BIST "
             "day after the announcement)",
         cols={"date": "trading day", "usdtry": "TCMB indicative USD/TRY buying rate [TRY]",
               "eurtry": "TCMB indicative EUR/TRY buying rate [TRY]", "policy_rate": "TCMB one-week repo rate [%]",
               "cpi_yoy": "latest published yearly CPI change (TÜİK) [%]", "cpi_mom": "latest published monthly CPI change [%]",
               "cpi_month": "reference month of the CPI value in use (YYYY-MM)",
               "fed_lower": "Fed funds target range, lower bound [%]", "fed_upper": "Fed funds target range, upper bound [%]",
               "usdtry_chg_20d": "20-day % change of USD/TRY [%]", "real_rate": "policy_rate − cpi_yoy [pt]",
               "rate_change": "change of policy_rate vs the previous trading day [pt]"}),
    dict(key="kap_dates", title="KAP publication dates of financial statements",
         file="reports/kap_disclosure_dates.csv", load=_csv("reports/kap_disclosure_dates.csv"), kind="structured",
         step="7B", grain="one row per stock and quarter (545)", pit="the anchor for all fundamental data", cols=KAP_DATES),
    dict(key="tech_events", title="Technical reversal events", file="reports/technical_events.parquet",
         load=_pq("reports/technical_events.parquet"), kind="structured (event table)", step="4",
         grain="one row per event; repeats of the same event within 5 days dropped",
         pit="event detected from data up to its date; forward returns are OUTCOMES, used only for research and "
             "for history whose window closed before an analysis date",
         cols={"date": "event date", "ticker": "stock code", "event": "event type (22 types, see Report 1; the harness uses the first 12, `HARNESS_EVENTS`)",
               "direction": "+1 bullish, −1 bearish", "regime": "`uptrend` close > SMA200, `downtrend`, `unknown`",
               "vol_confirm": "volume z-score > 1 on the event day", "quake_window": QUAKE,
               # Adım 2: olay anındaki gösterge durumları ve önceden yazılmış birleşik koşullar
               "rsi_state": "`dip` RSI(14) ≤ 35 on one of the last 3 days, `tepe` ≥ 65, `notr`, `bilinmiyor`",
               "kama_state": "5-day KAMA change: `yukselen` > +0.5%, `dusen` < −0.5%, `yatay` in between",
               "kama_turn": "5-day KAMA change minus the same change 5 days earlier [pt]; + = slope turning up",
               "st_state": "Supertrend direction at the event: `yukari`, `asagi`",
               "cloud_pos": "close vs Ichimoku cloud: `ustunde` above, `icinde` inside, `altinda` below",
               "vol_recent": "volume z-score > 1 on at least one of the last 3 days",
               "conf_rsi": "confirmation 1: RSI zone in the event's direction (bullish `dip`, bearish `tepe`)",
               "conf_vol": "confirmation 2: same as `vol_recent`",
               "conf_kama": "confirmation 3: KAMA turning in the event's direction (sign of `kama_turn`)",
               "conf_score": "number of confirmations, 0–3",
               "k1_confirmed": "pre-registered K1: at least 2 confirmations",
               "k1_all3": "all three confirmations (the spec's literal example; counted, not tested)",
               "k2_against_trend": "pre-registered K2: Supertrend against the event and price on the opposite side "
                                   "of the cloud (always false for Supertrend and cloud events)",
               **{k: v for k, v in EVENT_RET.items() if k[-2:].strip("_") in ("1", "3", "5", "10") and not k.startswith("gap")}}),
    dict(key="sector_events", title="Sector laggard / leader events", file="reports/sector_events.parquet",
         load=_pq("reports/sector_events.parquet"), kind="structured (event table)", step="5",
         grain="one row per trigger (first day the lag threshold is crossed; 10-day gap per stock)",
         pit="trigger uses past returns only; fwd/exc/gap columns are outcomes",
         cols={"date": "trigger date", "ticker": "stock code", "peer_group": "peer group",
               "config": "`lag10_5pct` (≥5 pt in 10 days) or `lag20_10pct` (≥10 pt in 20 days)",
               "side": "`laggard` (behind peers) or `leader` (ahead)", "lag": "stock return − peer median at trigger [fraction]",
               "shock": "a daily move of ≥ ±9.5% in the previous 10 days", "quake_window": QUAKE,
               **{k: v for k, v in EVENT_RET.items() if k.split("_")[1] in ("5", "10", "20")}}),
    dict(key="earnings_events", title="Earnings disclosure events", file="reports/earnings_events.parquet",
         load=_pq("reports/earnings_events.parquet"), kind="structured (event table)", step="7C",
         grain="one row per stock and quarter",
         pit="fundamentals as they could be known on `event_day` (src/data/pit_fundamentals.py): a value is used only "
             "after the version we hold was published, and only with inputs in the same unit. Inflation-accounting "
             "restatements of 2022Q4 – 2023Q3 were first published a year later, so those quarters have no values "
             "on their own event day (reason in `pit_dropped`); they serve only as next year's comparatives. TRY "
             "amounts are in the money of that report (`unit_flow`, `unit_stock`); windows are outcomes",
         cols={**KAP_DATES,
               "net_income": "quarterly net income attributable to parent, as published [TRY, unit in `unit_flow`]",
               "net_income_prev": "same quarter one year earlier as shown in this report (restated to its unit under "
                                  "TMS 29) [TRY]",
               "ni_yoy": "net income YoY change, both quarters in the same unit [fraction]",
               "ni_qoq": "net income QoQ change, both quarters in the same unit [fraction]",
               "loss": "True if net income < 0; empty if net income is not known on the event day",
               "net_income_ttm": "net income of the last 12 months, from cumulative figures [TRY]",
               "revenue": "net sales, quarterly [TRY]; missing for financials",
               "gross_profit": "gross profit, quarterly [TRY]; missing for financials",
               "op_income": "operating profit, quarterly [TRY]; missing for financials",
               "equity": "equity attributable to parent at quarter end [TRY, unit in `unit_stock`]",
               "total_assets": "total assets at quarter end [TRY, unit in `unit_stock`]",
               "roe_q": "quarterly net income / equity [fraction]", "leverage": "total assets / equity [ratio]",
               "net_margin": "net income / revenue, quarterly [fraction]; non-financials only",
               "revenue_yoy": "revenue YoY change [fraction]", "op_income_yoy": "operating profit YoY change [fraction]",
               "margin_chg": "net margin minus net margin one year earlier [fraction points]",
               "roe_chg": "roe_q minus roe_q one year earlier [fraction points]",
               "ebitda": "EBITDA of the quarter = gross profit − general & administrative − selling & marketing − R&D "
                         "expenses + depreciation (the definition companies publish) [TRY]; empty for financials and "
                         "KCHOL (a holding that consolidates a bank)",
               "ebitda_ttm": "EBITDA of the last 12 months [TRY]", "revenue_ttm": "revenue of the last 12 months [TRY]",
               "ocf_ttm": "operating cash flow of the last 12 months ('İşletme Faaliyetlerinden Nakit Akışları') [TRY]",
               "fcf_ttm": "free cash flow of the last 12 months = operating cash flow − purchases of fixed and "
                          "intangible assets [TRY]",
               "net_debt": "financial debt (short + long term) − cash − short-term financial investments [TRY]",
               "minority_interest": "minority (non-controlling) interests = total equity − parent equity [TRY]",
               "roa": "net income of the last 12 months / total assets at quarter end [fraction]",
               "roe_ttm": "net income of the last 12 months / parent equity at quarter end [fraction]",
               "ebitda_margin_ttm": "EBITDA / revenue, last 12 months [fraction]",
               "ebitda_margin_q": "EBITDA / revenue, quarter [fraction]",
               "ebitda_yoy": "EBITDA YoY change, both quarters in the same unit [fraction]",
               "nd_ebitda": "net debt / EBITDA of the last 12 months [ratio]; empty when EBITDA ≤ 0",
               "debt_to_equity": "financial debt / total equity (incl. minorities) [ratio]",
               "current_ratio": "current assets / current liabilities [ratio]",
               "quick_ratio": "(current assets − inventories) / current liabilities [ratio]",
               "cash_conversion": "operating cash flow / EBITDA, last 12 months [ratio]; empty when EBITDA ≤ 0",
               "ebitda_margin_chg": "quarterly EBITDA margin minus the same quarter a year earlier [fraction points]",
               "deleveraging": "net debt / EBITDA a year earlier minus now (positive = leverage fell) [ratio points]",
               "era_flow": "income-statement version: `A` as published (nominal), `B` inflation-accounting "
                           "restatement first published a year later, `C` inflation-accounting report",
               "era_stock": "balance-sheet version, same codes as `era_flow`",
               "unit_flow": "unit of income amounts: `TL nominal`, `TL@YYYY-MM` (money of that month) or `-` if none shown",
               "unit_stock": "unit of balance-sheet amounts, same codes as `unit_flow`",
               "pit_dropped": "values left empty and why: `gec(date)` known only from that date, `birim` inputs in "
                              "different units, `eksik` input missing, `karsilastirilamaz` comparison not in the same "
                              "unit, `anlamsiz` not meaningful (e.g. net debt / EBITDA when EBITDA ≤ 0)",
               "tercile": "`alt`/`orta`/`ust` = ni_yoy at or below the previous quarter's 1/3 point, up to its 2/3 "
                          "point, above it (no look-ahead: only previous-quarter values published by `event_day`)",
               "tercile_ref": "previous quarter whose values set the cut-offs",
               "tercile_n": "number of previous-quarter values used (at least 15, else no tercile)",
               "tercile_lo": "1/3 point of the previous quarter's ni_yoy [fraction]",
               "tercile_hi": "2/3 point of the previous quarter's ni_yoy [fraction]",
               "quake_window": QUAKE, **WIN, "financial": "True for AKBNK, ISMEN, TURSG (industrial ratios not used)"}),
    dict(key="news", title="News archive (KAP + RSS)", file="reports/news_events.parquet",
         load=_pq("reports/news_events.parquet"), kind="unstructured text + structured tags", step="10",
         grain="one row per disclosure/headline (KAP 2021-11 – 2026-11; RSS only from 2026-11-13)",
         pit="usable from `event_day`; `fetched_at` records when we downloaded it",
         cols={"source": "`KAP`, `AA`, `BloombergHT`, `TRT Haber`", "publish_time": "publication timestamp (Europe/Istanbul)",
               "event_day": EVENT_DAY, "ticker": "stock code (KAP: from the filer; RSS: company-name rules)",
               "sector": "sector of the ticker", "scope": "`sirket` company / `makro` macro / `genel` general",
               "category": "main keyword category (Turkish label, e.g. `borclanma` debt issuance)",
               "categories": "all matching categories, `|`-separated",
               "title": "**unstructured**: headline / KAP disclosure title (original text kept)",
               "summary": "**unstructured**: KAP summary line or RSS description (original text kept)",
               "url": "link to the original source", "fetched_at": "download time"}),
    dict(key="news_signals", title="KAP news signal events", file="reports/news_signal_events.parquet",
         load=_pq("reports/news_signal_events.parquet"), kind="structured (event table)", step="10C",
         grain="one row per stock, day and signal type (437)", pit="rules fixed on 2026-11-18 before outcomes were seen",
         cols={"ticker": "stock code", "event_day": EVENT_DAY, "publish_time": "KAP publication time",
               "direction": "`olumlu` positive / `olumsuz` negative",
               "kind": "signal type: yeni_is_sozlesme, geri_alim_baslangic, yonetici_ayrilis, bilanco_erteleme, dava_ceza, "
                       "tasfiye_iflas", "title": "**unstructured**: KAP title", "summary": "**unstructured**: KAP summary",
               "url": "KAP link", **WIN, "quake_window": QUAKE}),
    dict(key="speech_segments", title="Speech segments", file="reports/speech_segments.parquet",
         load=_pq("reports/speech_segments.parquet"), kind="unstructured text", step="10B",
         grain="one row per sentence/segment (Fed 129, TCMB 234)", pit="usable after `publish_time`", cols=SPEECH_META),
    dict(key="speech_features", title="Speech features and claim checks", file="reports/speech_features.parquet",
         load=_pq("reports/speech_features.parquet"), kind="structured features from unstructured text", step="10B",
         grain="one row per segment", pit="claims checked with data available at `publish_time`",
         cols={**SPEECH_META, "topics": "`|`-separated topics found by keyword rules (inflation, rates, growth, …)",
               "hawkish": "number of hawkish keyword hits in the sentence", "dovish": "number of dovish keyword hits",
               "uncertain": "number of uncertainty keyword hits", **CLAIM,
               "segment_type": "`dogrulanmis_olgu` verified fact / `sayisal_iddia` numerical claim / `yorum` commentary",
               "in_evidence": "True only for VERIFIED claims — the only speech content allowed into committed evidence"}),
    dict(key="speech_claims", title="Numerical claims in speeches", file="reports/speech_claims.csv",
         load=_csv("reports/speech_claims.csv"), kind="structured", step="10B", grain="one row per numerical claim (33)",
         pit="as above", cols={"source_id": SPEECH_META["source_id"], "segment_no": SPEECH_META["segment_no"],
                               "start": SPEECH_META["start"], "end": SPEECH_META["end"], **CLAIM,
                               "text": "**unstructured**: the sentence containing the claim"}),
    dict(key="bundle", title="Evidence bundle (one per stock and analysis day)", file="computed: "
         "src/harness/evidence.py run_analysis(ticker, asof)", load=_bundle, kind="structured + text evidence",
         step="11", grain="one bundle per analysis; stored as JSON in `analyses`",
         pit="everything is computed from data known at the analysis day's close; the quality gate checks "
             "`source_dates` ≤ asof",
         cols={"meta.ticker": "stock code", "meta.asof": "analysis day (last trading day ≤ requested date)",
               "meta.requested_asof": "date asked for", "meta.known_ticker": "False → HARD_BLOCK",
               "meta.sector": "sector", "observed.price.date": "last price date used",
               "observed.price.close": "close [TRY]", "observed.price.ret_5d": "5-day return [fraction]",
               "observed.price.ret_20d": "20-day return [fraction]", "observed.price.vol_20d": "annualised volatility",
               "observed.price.xu100_ret_20d": "XU100 20-day return [fraction]",
               "observed.price.rel_xu_10": "10-day return vs XU100", "observed.price.rel_xu_20": "20-day return vs XU100",
               "observed.price.max_abs_ret_250": "largest absolute daily move in 250 days",
               "observed.price.shock": "≥ ±9.5% day in the last 10 trading days",
               "observed.price.shock_day": "day of the largest move in the last 10 days",
               "observed.price.shock_ret": "signed return of that day [fraction]",
               **{f"observed.technical.{k}": f"value of feature `{k}` on the analysis day" for k in
                  ("rsi_14", "macd_hist", "bb_pctb", "bb_bandwidth", "st_dir", "kama_slope", "cloud_state", "vol_z_20",
                   "dist_support", "dist_resistance", "support_touches", "resistance_touches")},
               "observed.technical.above_kama": "close > KAMA", "observed.technical.above_sma_50": "close > SMA50",
               "observed.technical.above_sma_200": "close > SMA200",
               "observed.technical.recent_events": "technical events of the last 3 trading days (date, event, direction)",
               "observed.sector.peer_group": "peer group", "observed.sector.peer_size": "group size incl. the stock",
               **{f"observed.sector.sector_lag_{n}": f"feature `sector_lag_{n}`" for n in (5, 10, 20)},
               "observed.sector.sector_median_10": "peers' median 10-day return [fraction]",
               "observed.sector.rank_in_group": "rank of the stock's 10-day return in its group (1 = best)",
               "observed.sector.own_ret_10": "stock's 10-day return",
               "observed.fundamentals.period_end": "latest PUBLIC quarter", "observed.fundamentals.event_day": "its KAP event day",
               "observed.fundamentals.ni_yoy": "net income YoY (both quarters in the same unit)",
               "observed.fundamentals.tercile": "net income YoY third (cut-offs from the previous quarter, no look-ahead)",
               "observed.fundamentals.tercile_ref": "previous quarter that set the cut-offs",
               "observed.fundamentals.tercile_n": "number of previous-quarter values behind the cut-offs",
               "observed.fundamentals.tercile_lo": "1/3 cut-off [fraction]",
               "observed.fundamentals.tercile_hi": "2/3 cut-off [fraction]",
               "observed.fundamentals.loss": "loss quarter (empty if not known on the event day)",
               "observed.fundamentals.days_since": "trading days since event day",
               "observed.fundamentals.financial": "financial company",
               "observed.fundamentals.net_income": "quarter net income as published [TRY, money of that report]",
               "observed.fundamentals.net_income_prev": "same quarter a year earlier as shown in that report [TRY]",
               "observed.fundamentals.unit": "unit of the two amounts, e.g. `TL@2026-09` (money of Sep 2026)",
               "observed.fundamentals.roe_q": "quarterly ROE", "observed.fundamentals.leverage": "assets / equity",
               "observed.fundamentals.net_margin": "net margin",
               **{f"observed.macro.{k}": f"macro panel `{k}` on the analysis day" for k in
                  ("policy_rate", "cpi_yoy", "real_rate", "usdtry", "usdtry_chg_20d", "fed_lower", "fed_upper", "cpi_month")},
               "observed.macro.risk_half_weight": "macro risk flag (half position size in strategy D)",
               "observed.news.negative": "negative KAP signals of the last 10 trading days (day, kind, summary, url)",
               "observed.news.positive": "positive KAP signals of the last 10 trading days",
               "observed.news.kap_news_180d": "number of KAP disclosures in 180 days (coverage)",
               "observed.speech.sources": "official speeches of the last 30 days: speaker, time, stance, verified claims",
               "observed.speech.warnings": "transcript quality warnings",
               "history.catch_up": "past no-shock laggard outcomes closed before asof: n, hit, mean gap, last date",
               "history.fundamental": "past top-minus-bottom earnings reaction closed before asof",
               "history.tech:<event>": "past direction-adjusted 5-day hit rate vs XU100 of that event type",
               "source_dates.<source>": "latest date of each source used (checked by the gate)",
               **{f"conditions.{k}": v for k, v in {
                   "laggard": "sector_lag_10 ≤ −5 pt", "sector_strong": "peers' median 10-day return > 0",
                   "recovering": "close > KAMA or Supertrend up", "weak_earnings": "latest earnings in bottom third",
                   "catch_up": "laggard ∧ sector_strong ∧ recovering ∧ ¬weak_earnings",
                   "fundamental": "top third ∧ ¬loss ∧ ≤63 trading days since disclosure",
                   "technical": "bullish event in last 3 days and no bearish one",
                   "tech_conflict": "bullish and bearish events in the same 3 days", "shock": "price shock",
                   "negative_news": "negative KAP signal in last 10 days"}.items()},
               "gate.status": "PASS / PASS_WITH_WARNINGS / INVESTIGATE / ANALYSIS_UNSAFE / HARD_BLOCK",
               "gate.hard_blocks": "reasons for HARD_BLOCK (unknown ticker, future data, period mismatch)",
               "gate.unsafe": "reasons for ANALYSIS_UNSAFE (market closure)", "gate.soft_warnings": "soft warnings (3 → INVESTIGATE)",
               "backtest.variant": "strategy variant matching the action", "backtest.start": "backtest start",
               "backtest.end": "backtest end (≤ asof)", "backtest.toplam_getiri": "total return after costs [fraction]",
               "backtest.esit_agirlik": "equal-weight buy-and-hold [fraction]", "backtest.fark": "difference to equal weight",
               "backtest.xu100": "XU100 return [fraction]", "backtest.xu100_fark": "difference to XU100",
               "backtest.sharpe": "annualised Sharpe", "backtest.max_dusus": "maximum drawdown [fraction]",
               "backtest.islem_sayisi": "number of position entries", "backtest.uyari": "`az sinyal` if < 30 trades",
               "action": "rule-chosen label from the 8 allowed actions", "secondary": "other candidate labels that also held",
               "warning": "fixed disclaimer", "committed[].id": "evidence id E1, E2, …",
               "committed[].text": "**text generated from numbers by code** (not by the LLM)",
               "committed[].value": "the number/flag behind the sentence", "committed[].tool": "MCP tool that produces it"}),
    dict(key="analyses", title="Decision log: analyses", file="reports/decision_log.sqlite (table `analyses`)",
         load=_sql("analyses"), kind="structured (JSON fields)", step="11, 13", grain="one row per analysis",
         pit="written once; replay recomputes and compares",
         cols={"analysis_id": "12-character id", "created_at": "time of the run", "ticker": "stock code", "sector": "sector",
               "asof": "analysis day", "data_start": "first data date", "data_end": "last data date",
               "data_versions": "JSON: SHA-256 (12 chars) of each input file → replay detects changed data",
               "procedure_version": "hash of the workflow, action set and rules", "tools_called": "JSON list of MCP tools used",
               "observed": "JSON: observed context (see evidence bundle)", "history": "JSON", "backtest": "JSON",
               "conditions": "JSON", "gate_status": "gate status", "gate": "JSON gate result", "action": "final rule action",
               "secondary": "JSON", "committed": "JSON committed evidence", "explanation": "**unstructured**: LLM explanation",
               "fingerprint": "hash of the decision-relevant fields (replay equality check)"}),
    dict(key="reviews", title="Decision log: human reviews", file="reports/decision_log.sqlite (table `human_reviews`)",
         load=_sql("human_reviews"), kind="structured + text", step="11", grain="one row per review",
         pit="kept separate from the AI result: the system never turns an AI label into an order",
         cols={"review_id": "auto id", "analysis_id": "reviewed analysis", "reviewed_at": "time", "reviewer": "who",
               "decision": "ACCEPT / MODIFY / REJECT", "final_action": "new label if MODIFY",
               "reason": "**unstructured**: mandatory free-text reason"}),
    dict(key="traces", title="Decision log: harness traces", file="reports/decision_log.sqlite (table `harness_traces`)",
         load=_sql("harness_traces"), kind="structured (JSON)", step="13", grain="one row per harness run",
         pit="full state-machine trace for replay",
         cols={"analysis_id": "analysis", "created_at": "time", "llm": "model used, e.g. `gemini:gemini-3.1-flash-lite`",
               "summary": "JSON: tool calls, LLM calls, numbers checked, unsupported numbers",
               "trace": "JSON: states visited, every tool call and LLM call with validation result"}),
    dict(key="scan", title="Universe scan (spec §33)", file="reports/scans/scan_<asof>_<action>.csv",
         load=lambda: pd.read_csv(sorted(Path("reports/scans").glob("scan_*.csv"))[-1]), kind="structured",
         step="Adım 4 (22 Nov)", grain="one row per stock of the 30-stock universe on the scan day",
         pit="each row is the rule layer (`run_analysis`) on the scan day: only data public by that day",
         cols={"ticker": "stock code", "peer_group": "sector group", "action": "rule action on that day",
               "secondary": "other candidate labels, comma separated", "gate": "quality-gate status",
               "sector_lag_10": "10-day return minus peer median [fraction]",
               "sector_median_10": "peer median 10-day return [fraction]",
               "laggard": "sector_lag_10 at or below the catch-up threshold", "sector_strong": "peer median > 0",
               "recovering": "close above KAMA or Supertrend up", "weak_earnings": "latest statement in bottom third",
               "catch_up": "all catch-up conditions met", "fundamental": "fundamental-improvement conditions met",
               "technical": "bullish technical event without a bearish one", "tech_conflict": "bullish and bearish events",
               "shock": "large one-day move", "negative_news": "negative KAP news in the window"}),
    dict(key="scans", title="Decision log: universe scans", file="reports/decision_log.sqlite (table `scans`)",
         load=_scan_schema, kind="structured (JSON fields)", step="Adım 4 (22 Nov)", grain="one row per scan",
         pit="written once; `replay_scan` recomputes the scan and compares",
         cols={"scan_id": "12-character id", "created_at": "time of the scan", "asof": "scan day",
               "action": "question asked (candidate label)", "top_n": "maximum candidates sent to the full harness",
               "data_versions": "JSON: hash of each input file", "procedure_version": "hash of the workflow and rules",
               "rows": "JSON: the scan table", "candidates": "JSON: stocks whose action is the asked label",
               "also_secondary": "JSON: stocks where the asked label is only secondary (blocked or lower priority)",
               "near_misses": "JSON: lagging stocks in a strong sector that missed, with the failed conditions",
               "selected": "JSON: candidates sent to the full harness",
               "analysis_ids": "JSON: their decision-log ids (`analyses`)", "fingerprint": "hash for replay equality"}),
    dict(key="experiments", title="Harness experiment results", file="reports/experiments/raw.jsonl",
         load=_raw, kind="structured + text", step="14", grain="one row per case × version (A–E) × run",
         pit="each run uses only data up to the case date",
         cols={"ticker": "case stock", "asof": "case date", "label": "case description", "version": "A–E harness version",
               "run": "1 or 2", "model": "LLM used", "action": "action returned", "ref_action": "rule-based reference action",
               "valid_action": "action in the allowed set", "matches_reference": "action == reference",
               "unsupported_conclusion": "action not supported by deterministic evidence",
               "numbers": "numbers in the explanation", "fabricated": "numbers not found in tool outputs",
               "rounding": "numbers that are rounding mistakes", "evidence_completeness": "share of committed evidence covered",
               "tool_calls": "MCP tool calls", "invalid_calls": "calls rejected (unknown tool / bad arguments)",
               "tool_errors": "tool errors", "future_calls": "tool calls asking for data after asof",
               "llm_calls": "LLM calls", "latency": "seconds (unreliable: includes human time and cache)",
               "backtest_run": "a backtest was run", "explanation": "**unstructured**: final explanation text",
               "unsupported": "list of fabricated numbers", "rounding_list": "list of rounding mistakes",
               "review": "human review (E only)", "all_agents_hallucination": "fabricated numbers across all agent calls",
               "tools_used": "list of tools used"}),
]


def column_stats(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        s = df[c]
        nn = s.notna()
        if s.map(lambda v: isinstance(v, (list, dict, np.ndarray))).any():
            typ, rng = "list/JSON", "–"
        elif pd.api.types.is_bool_dtype(s):
            typ, rng = "bool", f"{s.mean() * 100:.0f}% True" if len(s) > 1 else str(s.iloc[0])
        elif pd.api.types.is_datetime64_any_dtype(s):
            typ = "datetime"
            rng = "–" if not nn.any() else (f"{s.min():%Y-%m-%d}" if len(s) == 1 else
                                             f"{s.min():%Y-%m-%d} – {s.max():%Y-%m-%d}")
        elif pd.api.types.is_numeric_dtype(s):
            typ = "int" if pd.api.types.is_integer_dtype(s) else "float"
            rng = (f"{s.min():.4g} – {s.max():.4g}" if len(s) > 1 else f"{s.iloc[0]:.4g}") if nn.any() else "–"
        else:
            typ = "text"
            ex = s.dropna().astype(str)
            rng = (ex.iloc[0][:40] + ("…" if len(ex.iloc[0]) > 40 else "")).replace("|", "/") if len(ex) else "–"
        rows.append(dict(column=f"`{c}`", type=typ, filled=f"{nn.mean() * 100:.0f}%" if len(s) > 1 else "–",
                         **{"range / example": rng}))
    return pd.DataFrame(rows)


def check(ds: dict, df: pd.DataFrame = None) -> tuple:
    """(açıklaması olmayan sütunlar, artık olmayan açıklamalar)."""
    df = ds["load"]() if df is None else df
    cols = set(df.columns)
    return sorted(cols - set(ds["cols"])), sorted(set(ds["cols"]) - cols)


def build() -> str:
    parts, toc, total = [], [], 0
    for i, ds in enumerate(DATASETS, 1):
        df = ds["load"]()
        missing, extra = check(ds, df)
        if missing or extra:
            raise ValueError(f"{ds['key']}: açıklamasız {missing}, fazla {extra}")
        st = column_stats(df)
        st.insert(1, "description", [ds["cols"][c.strip("`")] for c in st["column"]])
        total += len(st)
        toc.append({"#": i, "dataset": ds["title"], "kind": ds["kind"], "step": ds["step"], "columns": len(st)})
        parts.append(f"""## {i}. {ds['title']}

- **File:** `{ds['file']}`
- **Type:** {ds['kind']} · **Built in step:** {ds['step']} · **Grain:** {ds['grain']}
- **Point-in-time rule:** {ds['pit']}

{md_table(st, index=False)}
""")
    md = f"""# Data Dictionary

_{DISCLAIMER}_

Generated by `python -m src.reports.build_reports --only 0`. Descriptions are written in
`src/reports/data_dictionary.py`; types, fill rates and ranges are read from the actual files. A test fails if any
column lacks a description, so this dictionary cannot drift from the code. "Unstructured" marks free text
(news, speeches, LLM explanations, review reasons); all other fields are structured.

Units: `[fraction]` 0.05 = 5%; `[%]` already in percent; `[pt]` percentage points; `[TRY]` Turkish lira.
"Point-in-time rule" states from when a value may be used in a historical decision.

**{len(DATASETS)} datasets, {total} fields.**

{md_table(pd.DataFrame(toc), index=False)}

{chr(10).join(parts)}"""
    return str(write("00_data_dictionary.md", md))
