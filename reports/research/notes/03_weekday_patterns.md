# Interpretation notes for Report 3 (edited by hand; the build script inserts each block by its key)

<!-- summary -->
**Short answer.** A few calendar effects are statistically real in this sample but none of them is a usable
trading rule.

- **Wednesdays were weak and Thursdays strong** for the 30-stock portfolio.
  - Wednesday's weakness survives even when all 180 tests in this report are corrected together.
  - Thursday's strength is the only weekday effect that keeps its size in the second half.
  - Part of the Thursday effect comes from central-bank decision days, which fall on Thursdays in our data.
- **Trading it does not pay.** Buying on Wednesday and selling on Thursday would have earned +201% before costs,
  only +42% with a 0.15% one-way cost, and lost money at 0.30%. XU100 rose 647% over the same five years.
- **The multi-day patterns go the "wrong" way.** Classic reversal ideas ("three down days, then a rebound",
  "buy after a big drop") do not appear. Large down days are followed by *further* underperformance (−1.35 pt in
  5 days), and a rising Friday tends to be followed by a rising Monday. Short-term moves in BIST tend to
  **continue**, not reverse.
- **Most single-stock weekday "patterns" are noise.** 44 of 150 tests have p < 0.05, against 7.5 expected by
  chance. That excess is mostly the common Wednesday/Thursday effect, and only 6 survive correction.

<!-- weekday -->
**Reading Table 1 and Figure 1.**

- **Wednesday** averages −0.13%, 0.41 pt below the other days (p = 0.001; random relabelling gives
  p = 0.002). But its **median is +0.03%**: the typical Wednesday is flat, and a few very bad Wednesdays create
  the average.
- **Wednesday's weakness is concentrated in falling markets.** With XU100 below its 200-day average, the
  Wednesday mean is −0.96% (only about 14% of the valid days fall in that regime).
- **Thursday** is +0.34 pt above the other days, in both halves (+0.33 and +0.35). It is positive for all
  30 stocks (Table 5), and its median (+0.55%) is even higher than its mean. It is the most robust calendar
  observation in this report.
- **Monday, Tuesday and Friday** effects from the first half disappear in the second half, the typical fate
  of a calendar pattern.

<!-- ppk -->
TCMB's Monetary Policy Committee announces rate decisions on Thursdays. In our macro data, every policy-rate change
falls on a Thursday.

- **Rate-change Thursdays** averaged +1.29% (n = 25), against +0.38% for other Thursdays.
  - Hikes: +1.71%, n = 10.
  - Cuts: +0.95%, n = 16.
  - Both are positive. That suggests a relief effect: uncertainty is resolved on decision day, whatever the
    direction.
- **Without decision days** the Thursday premium shrinks from +0.34 to about +0.25 pt and is no longer
  significant (p = 0.054).
- **Fed decisions** are also announced on Wednesday evening (Türkiye time), after BIST closes, so BIST
  reacts on Thursday. Fed rate changes in our data are therefore also dated on Thursdays.

**Caveat:** our data only shows meetings where the rate *changed*. Meetings that held the rate are invisible, so
"other Thursdays" still contains some decision days. With the official TCMB meeting calendar this could be tested
properly. Here it stays a plausible explanation, not a proven one.

<!-- per_stock -->
Table 3 is the clearest multiple-testing lesson in the project:

- **Raw test:** 44 of 150 stock × weekday tests have p < 0.05, far above the 7.5 expected by chance.
- **After correction:** only 6 survive Benjamini–Hochberg and 2 survive Bonferroni (ARCLK Thursday, EREGL
  Wednesday).

The excess is mostly *not* stock-specific. Table 5 shows that Wednesday is negative and Thursday positive for
nearly all 30 stocks, so the per-stock tests mostly rediscover the same market-wide effect 30 times. The spec's
"a second stock may show the opposite pattern" is true on Monday (AKBNK +0.60 vs MPARK −0.34) and Friday (ECILC +0.71
vs TCELL −0.40). On those days the portfolio effect is weak and stock-level differences are what is left: likely
noise, since none of them keeps its sign in both halves at the portfolio level.

<!-- streaks -->
None of the 18 streak tests survives correction (lowest q = 0.15).

- **After up-streaks:** the point estimates are mostly negative (up3 at 5 days: −0.29 pt, up4: −0.33 pt), a mild
  hint that stocks that rose several days in a row lag the average afterwards.
- **After down-streaks:** close to zero.
- **Consistency:** the first half and second half often disagree in sign (down2, down3 and down4 at 3 days).

<!-- new -->
The spec asks about specific patterns. Their results:

- **"Post-large-down-day mean reversion."** The opposite holds. A stock that falls 5–9.5% in a day (excluding
  limit-down days) underperforms the other 29 stocks by 0.41 pt the next day and **1.35 pt over 5 days**
  (t = −4.6, q = 0.001). A big drop usually carries information (news, selling pressure) that is not fully priced
  in one day.
- **"Three consecutive down days followed by a two-day rebound."** No effect: +0.05 pt, p = 0.59.
- **"Mon–Wed negative, then Thu–Fri positive"** (the THYAO illustration). After such weeks, Thursday+Friday was
  +0.96 pt better than in other weeks. This happened only **28 times** in five years and is not significant
  (p = 0.11). A pattern with 28 occurrences cannot carry a strategy.
- **"End-of-week continuation vs reversal."** Continuation. Monday after a rising Friday is 1.12 pt better than
  Monday after a falling Friday (q = 0.002). News and sentiment over the weekend tend to extend Friday's
  direction.

<!-- costs -->
This is the "prediction ≠ profitable strategy" lesson in one table.

- **Before costs:** the Wednesday-buy / Thursday-sell rule is in the market only 20% of the days and still
  triples the money (Sharpe 1.79).
- **With costs:** every week it buys and sells the whole portfolio, 7,500 position entries in five years. A
  0.15% one-way cost removes most of the return (+42%) and 0.30% turns it into a loss (−33%).
- **Against holding XU100:** XU100 grew 7.5× over the same period.
- **Cash earns nothing in the backtest.** The engine gives 0% on the 80% of days out of the market. In reality
  TRY cash earned high interest in 2023–2026, which would help the rule, but the comparison with XU100 would
  still fail.

A statistically significant weekday difference is a fact about averages, not a trade.

<!-- multiple -->
- **Tests run.** In total 180 tests were run for this report (Table 9), and 52 had raw p < 0.05.
- **Correction.** After correcting all of them together, **9 survive**:
  - the portfolio Wednesday effect,
  - six single-stock weekday effects (mostly the same Wednesday/Thursday effect),
  - the large-down-day continuation at 5 days,
  - the Friday → Monday continuation.
- **Thursday.** The portfolio Thursday effect (p = 0.007) does *not* survive when everything is corrected
  together, even though it is the most stable over time.

Significance after correction and stability over time are different questions, and a pattern needs both before
it is taken seriously. Only Wednesday (weaker in the second half) comes close.

<!-- turkey -->
- **Central-bank Thursdays.** TCMB rate decisions (Thursday 14:00) and Fed decisions (announced after BIST's
  close on Wednesday) both land on Thursday trading. 2023–2024 had an unusually long series of large TCMB moves:
  hikes from 8.5% to 50% between June 2023 and March 2024, then cuts from December 2024. This explains why
  decision Thursdays matter so much in this sample.
- **High-inflation drift.** With strong nominal market growth, every weekday has a positive median except
  Tuesday. Weekday tests compare days *with each other*, so the drift cancels out, but it is one reason the
  Wednesday-buy rule loses so badly to simply holding the index.
- **Stress periods.** Wednesday's weakness is concentrated in XU100-down regimes: late 2024, March–June 2025
  and September–November 2026 (Report 1). Some of the worst Wednesdays fall in these sell-offs. We did not
  test this day by day, so the link is an observation, not a cause.

Verified from our data: the weekday of every policy-rate change. From general knowledge: the TCMB rate path and
the TCMB and Fed announcement times.

<!-- harness -->
The MCP tool `test_weekday_pattern` lets an agent test a weekday or streak pattern for one stock. It returns
the Welch t-test, the first-half vs second-half consistency and a cost note. Calendar patterns are **not** used to
choose any harness action. Given this report, that is deliberate:

- most patterns disappear out of sample;
- the strongest ones are market-wide calendar effects tied to policy days;
- even the best one does not survive trading costs.

If an agent finds a "pattern" for one stock, the harness treats it as context that needs the multiple-testing
warning, not as evidence for a candidate label.

<!-- limitations -->
- **Five years, one market.** About 240 observations per weekday. Weekday effects are known to come and go
  across decades.
- **Decision days only partly identified.** Only rate *changes* are identifiable. Hold decisions, TÜİK inflation
  days (the 3rd of each month) and options/futures expiry days were not tested as separate calendar events.
- **Sector and macro-event conditions not tested.** We condition only on the XU100 trend, not on sector
  direction or macro-event weeks.
- **Normal approximation.** Welch tests use it, but the permutation p-values agree closely, so this does not
  change conclusions.
- **Simplified backtest.** No interest on cash, a fixed cost, and no market-impact or bid-ask modelling for the
  open/close.

<!-- conclusion -->
Calendar patterns are where data mining is easiest. In this sample:

- Wednesday-weak / Thursday-strong is the one weekday effect worth mentioning. It is partly explained by
  central-bank decision days and is not tradable after costs.
- Single-stock weekday patterns are mostly the same market effect repeated, plus noise.
- Short-term multi-day behaviour is continuation, not the reversal the classic patterns assume.

The project treats these results as **descriptive context only**: none of them is allowed to drive an action label
in the harness.
