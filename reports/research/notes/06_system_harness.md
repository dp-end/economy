# Interpretation notes for Report 6 (edited by hand; the build script inserts each block by its key)

<!-- summary -->
**Short answer.**

- **Backtests.** No rule-based strategy variant beat a simple equal-weight buy-and-hold of the 30 stocks after
  costs. The best variant (C, fundamental filter) beat XU100 over the full period (+24 pt), but every variant lagged
  XU100 in the second half. The macro and news layers (D, E) lowered returns. The backtest engine passes its own
  look-ahead self-test.
- **Harness.** 15 deterministic MCP tools sit under a fixed 17-state machine with per-state tool permissions,
  a rule-based action, a quality gate, human review and a SQLite decision log.
- **Experiments.** The version comparison shows what each layer adds:
  - **Tools alone** (B) improve the action from 17% to 50% agreement with the evidence-based reference but leave
    unsupported labels.
  - **The state machine** (C) removes the unsupported labels and forces backtests.
  - **The evidence bundle and gate** (D, E) give the reference action with complete, number-safe explanations.
- **Numbers.** After fixing a validator false alarm, no version invented numbers. The differences are in **which
  action** is chosen and **how completely** the evidence is explained.
- **Replay.** Replaying the real decision log with today's code reproduces the same actions and gates and correctly
  reports what changed since the decisions were made: the procedure, and one data file (the earnings table rebuilt
  by the point-in-time fix of 22 November).

<!-- backtest -->
**Reading Tables 1–4 and Figure 1.**

- **Strong first half, weak second half.**
  - First half (2022-10 – 2024-11, the inflation rally): every variant except the rate filter beat XU100 by
    27–58 pt (the rate filter: −22 pt), but none beat equal-weight buy-and-hold.
  - Second half: every variant earned only +2% to +14% while XU100 rose 33% and equal-weight 48%.
  - A trend-following, long-only rule set that worked in a strong nominal bull market did not survive the
    tight-money period. This is the most important out-of-sample result of the project.
- **Layers do not add return.** Fundamentals (C vs B) helped: +25 pt, with a higher Sharpe (1.44 vs 1.29) and a
  smaller drawdown (−21% vs −24%). Before the point-in-time fix of 22 November (findings 9 and 10) this gain looked
  twice as large (+53 pt): part of it came from earnings information that was not yet public. Macro (D) and news
  (E) cut returns by 37–50 pt. The rate filter cut 103 pt and did not improve the drawdown. The only material
  benefit of D/E is a slightly lower volatility (about 20% vs 21%; the rate filter 17%).
- **Costs matter.** With about 600–700 position entries, cumulative costs are 15–21% of capital. Doubling the cost
  to 0.30% one-way removes 28–54 pt.
- **The engine is honest.**
  - A deliberately impossible "buy today's winners" signal returns an absurd 5 × 10¹¹ % when the one-day shift is
    removed. With the shift it returns +11%.
  - The shift is what prevents look-ahead, and a test (`tests/test_strategies.py`) enforces it.
  - Without costs, the shifted signal earns +1,075%. Yesterday's winners kept winning the next day, the same
    short-term continuation found in Report 3, but the daily turnover makes it worthless after costs.
- **Picking stocks inside sectors adds little (Table 3b).**
  - Against the sector-neutral benchmark (same sector weights, all sector members in equal weight), most variants
    are within ±13 pt over four years.
  - The sector catch-up variant B is **24 pt behind** it. Its laggards did not beat their own sectors over the
    holding period; whatever B earned came from the sectors it was in, not from the laggard choice. This fits
    Report 2: the harness rule's +0.8 pt edge is a 10-day effect, smaller than one round trip of costs, and B
    holds a stock for about 11 days.
- **Trades: fewer than half win (Table 3b).**
  - 44–49% of closed trades are positive after costs and the median trade loses 0.1–1.2%; only 39–47% beat XU100
    over the same days. The total return comes from a minority of large winners and from being invested in a
    rising market.
  - In the second half the winning share falls to 39–43%.
  - C, D and its two variants share the same trades: the macro filter only halves position sizes, which a trade's
    return does not see.
- **Rolling windows (Table 3c).** Each variant beat XU100 in 3–4 of the eight ~6-month windows and equal-weight
  buy-and-hold in only 2. The gap to XU100 swings from about −40 to +23 pt between windows.
- **Walk-forward selection does not help (Tables 3d, 3e).**
  - Choosing, every six months, the variant with the best Sharpe over the past year earned +52% over the
    three-year out-of-sample period, the same as XU100 (+53%) and below always holding C (+63%) or equal-weight
    buy-and-hold (+88%).
  - The past year's winner was the best variant of the next window only once in six. In May 2024 the rule picked
    the rate-filter variant, the weakest over the full period.
  - Its advantage over XU100 is a smaller drawdown (−20% vs −29%), not a higher return.
- **Spec checklist (§18).**
  - Done: signal shifting, publication dates, buy-and-hold, XU100 and sector-neutral benchmarks, return / Sharpe /
    drawdown / win rate (days and trades) / benchmark difference, costs, signal counts (no variant had fewer than
    30 entries), in-sample/out-of-sample halves, rolling windows and a walk-forward selection test.
  - Not done: the optional ML layer (§19).

<!-- harness -->
**Design choices, and why.**

1. **Tools compute, agents interpret.** Everything numerical (prices, indicators, events, statements, backtests,
   the gate) is a deterministic MCP tool with point-in-time inputs. Agents read tool results; they never compute.
2. **Fixed workflow.** The state order is code, not an LLM decision. A state can call only its listed tools. A
   blocked request is counted, not executed. Quality checks, the risk gate, human review and logging are mandatory,
   even on the safe-failure path (unknown ticker, earthquake closure).
3. **Rules choose, the LLM explains.** The action comes from `evidence.decide` over the 8 allowed labels and the gate
   status. The Orchestrator may only explain the **committed evidence** (E1, E2, …): sentences generated by code
   from numbers.
   - Every number in every agent's text is checked against the tool outputs.
   - An unsupported number triggers one correction request.
   - What remains is logged.
4. **Humans decide.** Nothing is ever turned into an order. The reviewer's decision and mandatory reason are stored
   next to the AI result, never merged into it.
5. **Memory is dated.** Episodic memory shows an agent at most three earlier cases of the same stock, and only cases
   whose analysis date is earlier than the current one.

<!-- scan -->
- **The scan answers "none" when that is the answer.** On the last data day (6 Nov 2026), during the fund-crisis
  sell-off, no stock is a catch-up candidate. Five stocks lag a strong sector, and the table says which condition
  stops each of them. For ECILC and EKGYO the only obstacle is a latest statement in the bottom third.
- **On 9 Jan 2026 there are four candidates.** The three with the largest sector lag (TTKOM, PGSUS, MAVI) went
  through the full harness with the mock model, and the scan and the three analyses were logged together (demo
  log). `replay_scan` recomputed the same 30 rows and the same candidates.
- **ARCLK on 9 Jan 2026** meets every catch-up condition but has bullish and bearish technical events in the same
  three days, so the rules say `INVESTIGATE`. The scan lists it as "secondary" with that reason instead of hiding it.
- **The scan adds no new decision rule.** Every row is exactly the rule layer's result for that stock and day (a
  test checks this), so the scan inherits the point-in-time guarantees and the evidence of Reports 1–5.
- **Gemini run pending.** The real-model run of a scan is part of the final experiment re-run.

<!-- experiments -->
**Status of this experiment.** It ran on 19–20 November 2026, before the point-in-time fix of 22 November (findings
9 and 10). We recomputed the rule result for all six cases with the corrected data. The action, the secondary
candidates and the gate are unchanged in every case, so the reference labels and the agreement figures below stand.
Some numbers in the evidence the models saw have changed:

| case | evidence item | in the experiment | now |
|---|---|---|---|
| ASELS, KRDMD (2026-11-06) | C backtest return (max drawdown) | 241% (−25%) | 213% (−21%) |
| ASELS, KRDMD | earnings-reaction history, top − bottom over days 0–4 | +1.97 pt (n = 162) | +2.59 pt (n = 131) |
| TURSG, PETKM (2026-11-06) | D backtest return (max drawdown) | 176% (−22%) | 163% (−21%) |
| PETKM | earnings third | top | middle (+48.1% vs cut-off +50.5%) |
| the four 2026 cases | earnings sentence | third within the quarter | third against the previous quarter's cut-offs; amounts with their unit |
| ASELS, 2023-02-15 | D backtest return | 38% | 41% |

The stored explanations therefore quote the pre-fix numbers. The experiment will be re-run once, after the remaining
corrections from the 22 November review, so that explanations and evidence match again.

**What each layer added** (Tables 7–8, Figure 2).

- **A — raw LLM.**
  - **Action:** correct in only 2 of 12 runs, only for the unknown ticker. Half of its labels are
    *unsupported*: a candidate label despite a shock, negative news or a missing condition.
  - **Context it missed:** it labelled TURSG (−39.5% in 20 days, negative KAP news) as `WATCH`, and called the
    earthquake case `WATCH` although the market was closed.
- **B — LLM + tools (free use).**
  - **Tool use:** only about 3 tool calls per run and never a backtest.
  - **Action:** right in 50% of runs, but it called TURSG a `TECHNICAL_REVERSAL_CANDIDATE` despite a shock and
    negative news, the most dangerous error in the table.
  - **Numbers:** after the validator fix (Section 5) it fabricated none and had small rounding errors (4.5% of
    numbers). Tools solved the *numbers* problem but not the *judgement* problem.
- **C — state machine.**
  - **Tool use:** forced to visit every state, it made about 25 tool calls and ran a backtest every time.
  - **Action:** no unsupported conclusions, but it still chose conservative labels in place of the evidence-based
    ones (`WATCH` for the ASELS earnings candidate, `INVESTIGATE` instead of `REJECT_SIGNAL` for the KRDMD shock).
  - **Errors:** 6% of its tool calls failed, mostly on the unknown ticker, where the tool errors were the
    *intended* behaviour.
- **D and E — evidence, gate, memory, human review.**
  - **Action:** 100% agreement with the reference. This is **by construction**: the reference is the rule result
    that D and E use. It shows the harness cannot drift from its rules, not that the rules are right (Reports 1–5
    discuss that).
  - **Explanations:** cover 97.5% of the committed evidence, against 38–50% for A–C, with no fabricated or rounded
    numbers. Semantic mistakes remain, as Section 5 shows.
- **Reproducibility** is 100% for every version. With temperature 0 and a separate cache for the second run, the
  same model chose the same action twice. That is consistency, not correctness: A was consistently wrong.
- **Cost.** D needs about 5 LLM calls and 23 tool calls per analysis. The time column is unreliable: it includes
  cached calls, retries after "503 high demand" errors and, for E run 1, the human's reading time.
- **Human review (Table 10).** The reviewer accepted 5 cases and modified one.
  - **KRDMD override:** `REJECT_SIGNAL` → `FUNDAMENTAL_IMPROVEMENT_CANDIDATE`, reason "it is in a position to
    rise". That reason is a forecast, and it contradicts evidence E4 (a −10% shock day and −15.9% vs XU100 −6.0% in
    20 days). The system allows the override, by design, and keeps the reason for later review.
  - **TURSG reason** ("moved in parallel with the market") also contradicts E2 (−39.5% vs −6.0%).
  - These two cases show why the log must store the human's reasoning, not only the final label: human judgement
    is also fallible and must be reviewable.

<!-- replay -->
**What the real log contains.** Two real Gemini runs (ASELS, 2026-11-06, made on 2026-11-18 during Step 13). In
both, the rules gave `FUNDAMENTAL_IMPROVEMENT_CANDIDATE`, and the human changed it to `WATCH`. Reason, in
translation: "strong earnings, but the strategy does not beat buy-and-hold; watch it". The `llm` column says only
`gemini` because these runs predate the model-name field. Newer runs store e.g. `gemini:gemini-3.1-flash-lite`.

**What the replay shows (Table 13).** Replaying both records today gives:

- the **same action and the same gate**;
- **one changed data file**: `reports/earnings_events.parquet`;
- **procedure changed = yes**, and a different fingerprint.

That is the correct diagnosis. Since 18 November the rules were extended (the conflicting-signal rule), the evidence
sentences were improved (shock direction, profit direction, units, earnings cut-offs), and on 22 November the
earnings table was rebuilt so that it contains only figures that were public on each event day (findings 9–10).
Neither change alters these two decisions. A replay separates "the data changed" from "our method changed". That is
the purpose of storing data hashes and a procedure hash with every decision.

<!-- findings -->
The checks built into the system (point-in-time gate, replay, numerical validator, tests) and the experiments
found problems 1–8 during Steps 14–15. Problems 9–10 were found on 22 November, while checking the project brief's
fundamental-analysis requirements against the data. Each was fixed with a test that fails without the fix:

| # | Problem | How it showed up | Fix | Effect on results |
|---|---|---|---|---|
| 1 | Episodic memory had no date filter | version E's 2023 case cited a decision made in 2026 | memory only shows cases with an earlier analysis date | ASELS 2023 E rows re-run |
| 2 | Gemini 3 tool calls need their `thought_signature` sent back | version B failed with HTTP 400 | assistant messages keep `extra_content` | none (crash fix) |
| 3 | Shock evidence had no direction | explanations could not say whether a −10% or +10% day happened | evidence states day, signed move, 20-day stock vs XU100 | D/E re-run for 3 cases |
| 4 | YoY profit % hid sign changes | PETKM "+48%" read as growth although still a loss | `profit_direction` words + both amounts | wording only, not re-run |
| 5 | Technical hit rate ignored direction for bearish events | `kama_turn_down` shown as 49% instead of 51%; PETKM's INVESTIGATE came from the bug | direction-adjusted rate; new conflicting-signal rule | PETKM/KRDMD D/E re-run |
| 6 | A speech published after the close entered that day's analysis | found while writing the data dictionary | speeches selected by event day | none (no case affected) |
| 7 | Validator flagged window lengths ("20, 50 and 200-day") as fabricated | B's "9.4% hallucination" was 6 such numbers | window/horizon lengths excluded like dates | stored explanations re-scored offline: hallucination 0% for all versions |
| 8 | `--mock` demo wrote to the real decision log | found while writing the README | mock runs use the demo database | none |
| 9 | Fintables stores inflation-accounting restatements and re-expressed figures, not the figures published at the time | 2023Q1–Q2 median profit growth was +178% and +209% for the 24 restating companies against +58% and +105% for the other six: restated figures, published a year later, were compared with original ones | point-in-time fundamentals layer: a figure is used only after the version we hold was published, comparisons only within one unit, amounts in the report's money; checked against 20 published figures | 2023Q1–Q3 no longer ranked; with row 10: Report 4's pre-event result and ROE drift no longer significant, C backtest 241% → 213% |
| 10 | Earnings thirds used companies that reported later | an early reporter's third depended on results published weeks later, and on 11 companies that reported after the data end | thirds from the previous quarter's cut-offs, using only figures already published; same-quarter ranking kept as a robustness check (Report 4) | C/D/E backtests, Reports 2 and 4, experiment evidence (Section 3) |

**Not caught automatically.** Some explanations cite the wrong evidence id. The KRDMD orchestrator attributed
technical hit rates to E1 (an earnings sentence) and skipped two evidence items in one run. The validator checks
numbers, not meaning. Catching such errors needs a second check (evidence-id consistency) or the human reviewer.

<!-- limitations -->
- **Small experiment.** 6 cases × 2 runs with one model at temperature 0. The cases were chosen to cover failure
  modes, not sampled randomly, so the percentages describe these cases, not general accuracy.
- **Circular reference.** The reference action is the rule result, so D/E agreement is guaranteed. A
  human-labelled reference set would measure whether the *rules* are good.
- **Mixed evidence bases.** Evidence completeness is a heuristic (evidence id cited, a number from the evidence, or
  a topic keyword). For PETKM and KRDMD, A–C were scored against the evidence list before the Step 15 rule changes,
  D/E against the new one. All stored explanations predate the point-in-time fix of 22 November (Section 3).
- **Backtest simplifications.** Halves instead of walk-forward, a fixed 0.15% cost instead of a slippage model, 0%
  interest on cash, weekly rebalancing only, no ML layer.
- **Short decision log.** It holds two real records. A longer history would make episodic memory meaningful.

<!-- conclusion -->
The system does what the project brief asks of an analytical harness:

- calculations are deterministic and point-in-time;
- the workflow and tool use are constrained;
- every conclusion is tied to committed evidence;
- numbers in explanations are checked;
- unsafe cases fail safely;
- humans decide;
- everything can be replayed.

The experiments show that the safety comes from the **structure** (state machine, rules, evidence, gate), not from
the model. The same LLM went from 17% to 100% agreement with the evidence-based answer as layers were added. The
backtests show the other half of the lesson: a careful, honest process mostly reports that the tested strategies do
**not** beat buy-and-hold after costs. The harness is built to report exactly that kind of result.
