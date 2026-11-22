# Interpretation notes for Report 1 (edited by hand; the build script inserts each block by its key)

<!-- summary -->
**Short answer.** After all 12 event types, the stock did move in the "expected" direction in raw terms, but
almost exactly as much as any random stock-day would have, because BIST 100 rose strongly in nominal TRY over
the sample. Measured against XU100, the average 5-day effect is between −0.13% and +0.32%, no event type
survives multiple-testing correction, and only 4 of 12 keep the same sign in the second half of the sample.
Two bullish events (`kama_turn_up`, `resistance_break`) are the most stable, but their edge (~+0.2–0.3% in 5 days)
is about the size of one round-trip trading cost (0.30% in our backtest).

The spec's combined conditions do not change this. Neither "confirmed" events (at least two of RSI zone, recent
volume spike and a turning KAMA) nor events "against the trend" (below a bearish Supertrend and the cloud) behave
reliably differently from the same event without the condition (20 pre-registered tests, none with q < 0.05).

<!-- main -->
**Reading Table 2 and Figure 1.**

- Bullish events show a raw 5-day gain of +0.95% to +1.26% and a positive outcome in 54–56% of cases. The
  any-day baseline is +1.08% and 55.1%. So the raw "success" is mostly **market drift**, not the signal.
- Bearish events look like failures in raw terms (raw hit 42–49%): stocks kept rising after a "sell" signal
  simply because the whole market was rising.
- Once XU100 is subtracted (orange bars), every event type is within ±0.35%. The lowest day-clustered p-value
  (`cloud_break_down`, p = 0.042) becomes q = 0.51 after correction for 12 tests.
- Clustering matters: `kama_turn_up` and `resistance_break` have event-level t = 2.30 and 2.53, but many of these
  events happen on the same days across stocks. Averaging per day first cuts t to 1.18 and 1.40. The plain t-test
  overstates confidence.

<!-- bounce -->
**Bounce counts look good; the typical event does not.** Lower-band re-entries bounced 347 times and failed
279 times, which reads like a usable reversal signal. But the **median** excess return of every bullish event is
negative (−0.08% to −0.56%) while the mean is slightly positive. The typical bullish event slightly
underperformed XU100, and a minority of large winners pulled the average up. A rule that trades every event
collects mostly the typical case.

<!-- horizons -->
Effects grow slowly with the horizon for the two stable bullish events (`kama_turn_up` +0.09% → +0.33%,
`resistance_break` +0.13% → +0.37% from 1 to 10 days), which looks more like mild trend continuation than a
sharp reversal. Bearish band/RSI exits (`bb_upper_reentry`, `rsi_overbought_exit`) get *worse* at 10 days
(−0.58%, −0.51%): after an overbought exit, the stock tended to keep outperforming the index. That is momentum,
the opposite of the reversal the event name suggests.

<!-- regimes -->
**Regimes change the answer, which is a warning, not a discovery.**

- Mean-reversion events work better in **high volatility**: `bb_lower_reentry` +0.72% (n = 193) and
  `rsi_oversold_exit` +1.02% (n = 79), against −0.22% and −0.28% in low volatility. This matches the spec's
  ASELS illustration, but the samples are small and these cells are not corrected for the ~70 regime tests.
- Bearish trend signals fail when the index is already falling: `supertrend_down` −1.40% (n = 105) in XU100-down
  periods. After a sell flip in a falling market, the stock fell *less* than the index.
- Cells with n < 50 (for example `rsi_overbought_exit` below its SMA200, +1.51% with n = 25) should be read as
  noise.

<!-- volume -->
Volume confirmation helps the breakout-type events a little: `resistance_break` +0.15% → +0.41%,
`cloud_break_down` +0.16% → +0.66%, `kama_turn_up` +0.23% → +0.35%. It hurts the bearish exhaustion events
(`rsi_overbought_exit` +0.01% → −0.38%, `supertrend_down` +0.03% → −0.27%). In other words, high volume tends to
mean "the move continues". These differences were not tested for significance and are a hypothesis for later
work, not a rule.

<!-- oos -->
Only `kama_turn_up`, `resistance_break`, `rsi_oversold_exit` and `bb_upper_reentry` keep their sign in both
halves. The largest first-half effect (`cloud_break_down` +1.19%) turns into −0.21%. The two halves are very
different markets (see Türkiye context): the first is the negative-real-rate boom, the second is the tight-policy
period with two political/financial shocks. A sign flip across them is what we would expect from a pattern that
is regime-specific or accidental.

<!-- multiple -->
None of the 32 tests has q < 0.05. The smallest p-values belong to the combined conditions (0.011 for
confirmed `bb_lower_reentry`), not to the plain events.

Read as a whole, the evidence is **"no reliable reversal edge beyond the market"**. If we had picked the best
cell in the regime table (+1.51%) and reported it alone, it would look like a strong finding. That is exactly the
data-snooping risk the project brief warns about.

<!-- turkey -->
The sample (Nov 2021 – Nov 2026) covers very different Turkish market regimes. These dates explain much of the
drift and the regime and half-sample differences above:

- **Dec 2021 – 2022: currency crisis and negative real rates.** The lira crisis of December 2021 and the
  FX-protected deposit scheme were followed by a year of high inflation (CPI peaked above 80% in late 2022) with a
  low policy rate. Equities became an inflation hedge and XU100 rose very strongly in nominal TRY. This is the main
  source of the positive any-day baseline.
- **Feb 2023: Kahramanmaraş earthquakes.** Borsa İstanbul suspended trading 8–14 Feb 2023. This window is
  excluded. XU100 20-day volatility peaked in Q1 2023 (annualised ≈ 46%).
- **Jun 2023 – Mar 2024: monetary tightening.** After the May 2023 elections the policy rate was raised step by
  step to 50%. The sample split (12 May 2024) falls just after this, so the second half is the tight-money market.
- **Mar–Jun 2025: political shock.** On 19 March 2025 the detention of İstanbul's mayor caused two
  circuit-breaker halts. The index lost about 16% in a week, the worst week since 2008, and the central bank
  tightened (overnight lending rate 46%). Our data shows XU100 below its 200-day average for most of
  Mar–Jun 2025.
- **Sep–Nov 2026: fund crisis.** On 17 Sep 2026 the Capital Markets Board (SPK) ordered the liquidation of 131
  funds. Financial press reports a ~25% fall in BIST 100 between 31 Aug and 30 Oct 2026, the sharpest two-month fall
  since 2008, together with an MSCI decision on 15 Oct to freeze some Türkiye index changes. In our data
  XU100 drops below its 200-day average on 18 Sep 2026, the next trading day, and stays there until the data ends
  (6 Nov 2026).

Verified by search: the March 2025 and Sep–Oct 2026 events (news sources listed in the final report). From
general knowledge, not re-checked here: the 2021–2024 items. The 2026 figures come from press reports that
differ slightly from one another.

<!-- harness -->
The harness (Step 11–13) does not trust a technical event by itself:

- A bullish event in the last 3 trading days, with no bearish event alongside it, can make the stock a
  `TECHNICAL_REVERSAL_CANDIDATE`.
- Every recent event carries its **historical** direction-adjusted 5-day hit rate vs XU100. Only events whose
  5-day window had already closed on the analysis date are used. A rate below 50% adds a "not better than a coin
  flip" soft warning; three soft warnings turn the quality gate to `INVESTIGATE`.
- If bullish and bearish events appear in the same 3-day window (**conflicting signals**), the action is
  `INVESTIGATE`. A candidate with a shock or negative news is still `REJECT_SIGNAL`.

**Bug found while writing this report.** The harness hit rate ignored the event direction, so for bearish events
it counted "stock beat XU100", which is the *failure* case. Example: `kama_turn_down` was shown as 49% instead
of 51%. In the PETKM experiment case, two of its three coin-flip warnings came from this bug, and they alone
produced `INVESTIGATE`. After the fix (direction-adjusted rate plus a test that fails without it), PETKM would have
fallen to `WATCH`. The conflicting-signal rule above was added so that PETKM is `INVESTIGATE` for the right reason.
The PETKM and KRDMD D/E experiment rows were re-run with the corrected code.

<!-- limitations -->
- **Missing event families.** The spec's middle-Bollinger-band test and support/resistance *touches* (as opposed
  to breaks) are not stored as events. RSI local troughs/peaks are approximated by exits from the 30/70 zones, and
  divergence is not detected.
- **Combined conditions are our definition.** The spec's literal triple (RSI trough + rising volume + flattening
  KAMA at once) is too rare to test, so K1 needs two of three confirmations, each judged over the last 3 days. The
  RSI (35/65), KAMA (±0.5%) and 3-day choices were fixed from event counts before returns were computed, and not
  varied afterwards. For the two RSI events almost every event is "against the trend", so their K2 comparison
  group is tiny (73 and 12 events).
- **No maximum-excursion measure.** Reversal strength is the median return at fixed horizons, not the largest
  favourable move inside the window.
- **Only nominal TRY returns.** High inflation inflates raw returns. The excess-vs-XU100 column is the fair
  comparison.
- **Overlapping windows.** 5-day windows overlap and events cluster on market-wide days. Day-clustering reduces
  but does not remove this dependence.
- **Arbitrary choices.** Regime thresholds (200-day SMA, expanding-median volatility) and the 5-day dedupe gap
  are fixed choices and were not optimised. Other choices could move individual cells.
- **Survivorship.** The universe is the 2026 list of 30 stocks, so firms that left the index are not included.

<!-- conclusion -->
Technical reversal events are useful as **timestamped context**: they say what the chart is doing. On these 30
stocks over five years, they do not give a reliable edge over simply holding the index once market drift,
clustering, multiple testing and a later sample are taken into account. This is the "prediction ≠ profitable
strategy" distinction in practice: an event can be right more than half the time in raw terms and still add
nothing after the benchmark and costs. Requiring confirming indicators, as the spec's ASELS example suggests,
does not rescue them on this data. The harness therefore uses these events only together with sector,
fundamental and risk-gate evidence, never alone.

<!-- combined -->
**Reading Table 6a.** The states show what the chart typically looks like at each event.

- **Reversal events happen against the trend almost by definition.** 91% of `bb_lower_reentry` and 100% of
  `rsi_oversold_exit` events come with a bearish Supertrend; their mirror events look the same.
- **KAMA rarely turns at a band re-entry** (5–7%). KAMA reacts slowly, so it turns days after the band touch,
  which is why the spec's triple occurs so rarely.
- **The "all three" cells are too small to read** (1–102 events). Examples are +4.24% for `supertrend_up` (n = 6)
  and +2.77% for `bb_lower_reentry` (n = 7).

**Reading Table 6b: no pre-registered condition is confirmed.**

- **K1 "confirmed".**
  - The closest case is `bb_lower_reentry`: +0.80% vs −0.16% without confirmation (difference +0.96 pt, p = 0.011,
    q = 0.34). The difference has the same size in both halves (+0.87 / +0.92).
  - It is the one cell that matches the spec's ASELS story, but it does not survive correction.
  - For the other 11 event types the difference is between −0.66 and +0.40 pt, five of them with the wrong sign.
- **K2 "against the trend".**
  - The expectation (weaker follow-through) holds for the bearish band and RSI events, e.g. `bb_upper_reentry`
    −0.63 pt and `rsi_overbought_exit` −1.62 pt (p = 0.045, q = 0.48).
  - For those RSI events the comparison group is tiny (73 events), and the `bb_upper_reentry` effect appears only
    in the first half (−1.38 / +0.23).
  - For `bb_lower_reentry` the sign is the opposite of the spec's example: a bullish re-entry below a bearish
    Supertrend and the cloud did *better* (+0.67 pt), in both halves. Deep oversold bounces are a mean-reversion
    trade, not a trend trade.

The ASELS example in the spec is a plausible story, and one cell (`bb_lower_reentry` with confirmations) points
in its direction. On 30 stocks over five years it is not a finding: 20 tests, none with q < 0.05.
