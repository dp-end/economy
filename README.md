# BIST 100 Agentic Analytics Harness — CSE-481 Engineering Economics Project

An educational research system for 30 BIST 100 stocks (Nov 2021 – Nov 2026). It builds point-in-time datasets
(prices, quarterly fundamentals, macro, KAP news, central-bank speeches), runs event studies and backtests, and
wraps the deterministic tools in an **MCP server** and a **stateful agent harness**. In the harness, rules choose one
of 8 analytical labels, an LLM (Gemini) only explains the committed evidence, and a human accepts, modifies or
rejects the result. Every run is logged for replay.

> **Not investment advice.** The system returns an evidence-backed *analytical state*, never a buy/sell order.
> There is no broker connection. Returns are nominal TRY unless stated otherwise.

Project brief: *Engineering Economics Project 2026, CSE-481* (Dr. Alper Özcan).

---

## 1. Quick start (about 10 minutes, no API key needed)

```bash
# 1) Python 3.12 and ffmpeg (ffmpeg only for re-transcribing speech audio)
python3.12 --version
brew install ffmpeg            # macOS; Linux: apt install ffmpeg

# 2) Code and environment
git clone https://github.com/dp-end/economy.git
cd economy
python3.12 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3) Data cache (delivered separately as economy_data_2026-11-06.zip, ~10 MB)
unzip /path/to/economy_data_2026-11-06.zip      # creates ./data/

# 4) Check
pytest -q                                       # expected: all tests pass (≈ 1 minute)
cp .env.example .env                            # keys are optional, see section 4
```

Then try the demos in section 3.

## 2. What is where

```
config/universe.csv        30 study stocks + 4 extra peers, sector and peer-group map (Step 1)
src/data/                  data loaders with on-disk caches: prices, fundamentals, KAP dates, macro, news, speech
src/indicators/            technical indicators, support/resistance, sector-relative features
src/events/                event studies: technical reversals, sector catch-up, calendar, earnings, news
src/backtest/              backtest engine (signal shift, costs, benchmarks) and strategy variants A–E
src/mcp_server/            MCP server with 15 deterministic tools + stdio client demo
src/harness/               evidence bundle, quality gate, decision log + replay, state machine, experiments
src/agents/                Gemini client (OpenAI-compatible endpoint), prompts, numerical validator, mock LLM
src/reports/               research reports, data dictionary, PDF builder
reports/                   committed outputs (event tables, summaries, backtests, decision logs, experiments)
reports/research/          final research reports (Markdown + PDF) and their interpretation notes
tests/                     pytest suite (offline; uses the data cache)
data/                      cache, NOT in git (see section 5)
```

Main documents:

- `reports/research/research_reports.pdf`: data dictionary plus the five research reports.
- `reports/research/00_data_dictionary.md`: every structured and unstructured field.
- `reports/research/01_…05_*.md`: technical reversals, sector catch-up, weekday patterns, earnings reaction,
  macro/news/speech context.

## 3. Demos (no API key)

```bash
# Full harness on one case with the mock LLM. Writes only to reports/decision_log_demo.sqlite.
python -m src.harness.state_machine ASELS 2026-11-06 --mock --review ACCEPT --reason "demo"

# Safe-failure cases
python -m src.harness.state_machine ASELS 2023-02-15 --mock --review ACCEPT --reason "earthquake"   # ANALYSIS_UNSAFE
python -m src.harness.state_machine YOKBOYLE 2026-11-06 --mock --review ACCEPT --reason "unknown"   # HARD_BLOCK

# Universe scan (spec §33): rule layer on all 30 stocks, no LLM; then the top 3 catch-up candidates
# through the full harness with the mock LLM. Writes to reports/decision_log_demo.sqlite and reports/scans/.
python -m src.harness.scan 2026-11-06 --demo-db                                  # no candidates that day
python -m src.harness.scan 2026-01-09 --run --mock --review ACCEPT --reason "demo"  # 4 candidates, 3 analysed
python -m src.harness.scan --replay <scan_id> --demo-db
# other questions: --action TECHNICAL_REVERSAL_CANDIDATE | FUNDAMENTAL_IMPROVEMENT_CANDIDATE

# Decision log, replay and episodic memory (demo database)
python -m src.harness.decision_log

# MCP: start the server as a subprocess (stdio), list the 15 tools, call each once
python -m src.mcp_server.client_demo

# Rebuild the research reports and the PDF (needs Google Chrome for the PDF step)
python -m src.reports.build_reports
python -m src.reports.pdf
```

The 8 actions: `HOLD`, `WATCH`, `INVESTIGATE`, `POTENTIAL_CATCH_UP_CANDIDATE`, `TECHNICAL_REVERSAL_CANDIDATE`,
`FUNDAMENTAL_IMPROVEMENT_CANDIDATE`, `REJECT_SIGNAL`, `ANALYSIS_UNSAFE`.

## 4. Runs that need API keys

Put the keys in `.env` (copy `.env.example`). `.env` is git-ignored; never commit it.

| Key | Used for | Needed when |
|---|---|---|
| `GEMINI_API_KEY`, `GEMINI_MODEL` | LLM agents (Step 13) and harness experiments (Step 14) | real LLM runs only; all tests use the mock |
| `EVDS_API_KEY` | TCMB EVDS macro series (Step 8) | only to re-download macro data |

```bash
# Real harness run with Gemini; asks for a human review in the terminal; writes to reports/decision_log.sqlite
python -m src.harness.state_machine ASELS 2026-11-06

# Universe scan with Gemini for the selected candidates (one human review per candidate)
python -m src.harness.scan 2026-01-09 --run

# Harness experiments A–E (6 cases × 2 runs, ~200 Gemini calls, resumable).
# Results are already committed in reports/experiments/ and reports/harness_experiments.csv.
python -m src.harness.experiments
```

> **Warning:** `python -m src.harness.experiments --fresh` deletes the committed experiment results. `--mock`
> appends to the same `raw.jsonl`. Use both only in a copy of the repository.

The experiments in the report were run with `GEMINI_MODEL=gemini-3.1-flash-lite`.

## 5. Data: archive or rebuild from sources

`data/` is not in git. It contains third-party data (KAP, İş Yatırım, Fintables, TCMB) that is shared only for
course evaluation. There are two options:

**A. Archive (recommended).** Unzip `economy_data_2026-11-06.zip` in the repository root. The maintainer creates it
with `python -m src.data.pack_data` (raw speech audio is excluded; transcripts and official texts are included).

**B. Rebuild from the original sources** (≈ 1 hour; run from the repository root, in this order):

| Step | Command | Source / note |
|---|---|---|
| 2 | `python -m src.data.prices` | İş Yatırım daily prices (public endpoint), quality report → `reports/price_quality.csv` |
| 2 | `python -m src.data.yahoo_prices` | Yahoo Finance, validation only |
| 2 | *manual* | Fintables quarterly statements: download `<TICKER> (TRY).xlsx` for the 30 stocks into `data/fintables/` (needs Fintables access); check with `python -m src.data.fundamentals` |
| 7 | `python -m src.data.disclosure_dates` then `python -m src.data.kap_dates` | legal filing windows, then real KAP publication times (public KAP API, polite 4 s pauses) |
| 8 | `python -m src.data.macro` | TCMB EVDS (needs `EVDS_API_KEY`); Fed table is in the code |
| 10 | `python -m src.data.kap_news` | KAP material-event disclosures 2021–2026 (~630 requests, ~45 min) |
| 10 | `python -m src.data.rss_news` | AA / BloombergHT / TRT RSS; archive grows from the day you start |
| 10 | `python -m src.data.news_tags` | keyword tagging → `reports/news_events.parquet` |
| 10B | `python -m src.data.speech` then `python -m src.data.speech_features` | Fed audio + Whisper (needs ffmpeg), TCMB official text; claim verification |
| 1d | `python -m src.data.market_cap` | İş Yatırım daily market capitalisation (150 requests); checks and aligns capital changes |

Then rebuild the research outputs:

```bash
python -m src.events.technical_events      # Step 4
python -m src.events.sector_catchup        # Step 5
python -m src.events.calendar_patterns     # Step 6
python -m src.events.earnings_reaction     # Step 7C (point-in-time fundamentals, src/data/pit_fundamentals.py)
python -m src.data.valuation               # daily P/E, P/B, EV/EBITDA -> reports/valuation_daily.parquet
python -m src.events.news_reaction         # Step 10C
python -m src.backtest.engine              # Step 9: engine checks (look-ahead test, costs)
python -m src.backtest.strategies          # Step 9–10: strategy variants A–E vs benchmarks
python -m src.reports.build_reports && python -m src.reports.pdf
```

Prices end on **2026-11-06**. Later data would change the numbers in the reports. Re-running the downloads on
another day gives a longer sample.

## 6. Point-in-time rules (no look-ahead)

- **Fundamentals:** usable only from the KAP **event day**, the first trading day after the real publication time
  (88% of filings are published after 18:00). Legal deadlines are not used.
- **Fundamental versions and units** (`src/data/pit_fundamentals.py`): Fintables keeps the latest version of each
  figure. For the 24 companies that apply inflation accounting (TMS 29) it shows the 2022Q4 – 2023Q3 restatements,
  which were published a year later, and re-expresses later figures in the money of the latest balance sheet
  (checked against 20 published figures; `python -m src.data.pit_fundamentals` prints the evidence). A figure is
  used only from the day its version was published, comparisons stay within one unit, and amounts are shown in the
  money of their report.
- **Earnings thirds:** a company's third is decided against the previous quarter's cut-offs, using only figures
  already published on its event day.
- **Valuation:** market capitalisation at that day's close and the latest statement public that day;
  inflation-accounting earnings, EBITDA and equity are moved to the money of the latest CPI published that day.
- **Macro:** CPI from the 3rd of the following month; Fed decisions from the first BIST day after the
  announcement; FX and policy rate at that day's close.
- **News and speeches:** usable from their event day. A speech published after the close is not part of that day's
  analysis.
- **Technical events:** use only past closes. Historical hit rates use only events whose forward window had closed
  before the analysis date.
- **Backtests:** shift signals by one day and charge 0.15% one-way costs.
- **Quality gate:** checks every source date against the analysis date and blocks the analysis if any is later.

The tests enforce these rules: `tests/test_harness.py`, `tests/test_strategies.py`, `tests/test_kap_dates.py`,
`tests/test_pit_fundamentals.py`, `tests/test_earnings.py` and others.

## 7. Tests

```bash
pytest -q            # whole suite, offline, uses data/ and reports/
pytest -q tests/test_harness.py tests/test_state_machine.py tests/test_mcp.py   # harness only
```

No test calls a paid API or uses a key. LLM behaviour is tested with `MockLLM`.

## 8. Known limitations

- **Sample:** 30 stocks, five years, nominal TRY. Survivorship: the 2026 index list is used for the whole period.
- **Fundamentals data:** around the inflation-accounting switch (2022Q4 – 2024Q1) earnings cannot be ranked,
  because the restated figures were published a year later. EBITDA, leverage, liquidity, cash-flow and return
  ratios are computed (`src/data/pit_fundamentals.py`; EBITDA, operating cash flow and net debt checked against
  published company figures) and daily valuation ratios (P/E, P/B, EV/EBITDA) are computed. Report 4 tests four
  of them as hypotheses written before testing (EBITDA growth, EBITDA-margin change, deleveraging, cash conversion;
  only the margin change is confirmed); liquidity ratios and EV/EBITDA are not tested. P/E cannot be computed for the 24 inflation-accounting companies from
  about May 2023 to March 2024, because the unrestated profits the market saw then are not in Fintables. Fintables
  keeps only the latest version of each figure, so later corrections are carried back into earlier periods (one
  small case found).
- **News:** RSS has no history, so news statistics rely on KAP. Categories come from keyword rules.
- **Speeches:** only two official speeches (no instructor-approved list arrived). The TCMB YouTube video required a
  bot sign-in that we did not bypass, so the official text was used instead.
- **Validator:** the numerical-hallucination check catches invented numbers but not wrong interpretations.

See section "Limitations" in each research report.
