# Interpretation notes for Report 2 (edited by hand; the build script inserts each block by its key)

<!-- summary -->
**Short answer.** There is a small but real catch-up effect. A stock that falls at least 5 points behind its
peers in 10 days, without a price shock, beats its peers by +0.59 pt on average over the next 10 days. A random
stock-day gives +0.15. This is the only result in the report that survives multiple-testing correction
(q = 0.041). The harness's extra conditions (strong sector, price recovering) raise the effect to about +0.8–0.9 pt,
and only the full harness rule keeps most of its size in the second half of the sample (+0.88 → +0.72, while the
other laggards fall from +1.00 to +0.02).

The effect is small and noisy:

- About half of the laggards (48–50%) are still behind their peers 20 days later.
- About 29% fall more than 5 points further behind.
- The long-only strategy built on the idea (B_sektor) roughly matches XU100 over the whole period (−1 pt), beats
  it in the first half and lags it in the second half.

<!-- main -->
**Reading Table 1.**

- **No-shock laggards** (`lag10_5pct`): +0.26 / +0.59 / +0.75 pt at 5 / 10 / 20 days, hit rate 52.1%. The
  baseline is +0.09 / +0.15 / +0.34, so the extra catch-up is roughly +0.4 pt per 10 days.
- **Laggards with a shock** behave the other way: −1.82 pt at 5 days, and they keep lagging. A ±10%-limit day
  usually means news (earnings, a capital increase, a legal problem). This is why the harness treats a shock as a
  blocker (`REJECT_SIGNAL`), not as a bargain.
- **The slower setting** (`lag20_10pct`, 10 pt in 20 days) is weaker (+0.37 pt, q = 0.73). A deeper, longer
  lag is more often a real change than noise.
- **Leaders do not reverse.** Their gap stays near zero or slightly positive (+0.29 at 10 days, not significant).
  Being ahead of the sector is not a reason to expect a fall.

<!-- ladder -->
- **Shock filter:** removing shock cases is the single most useful step (+0.41 → +0.59).
- **Sector strength and price recovery:** requiring both pushes the 10-day gap to +0.88 pt (hit rate 54.4%).
- **Earnings filter:** the last harness filter (drop laggards whose latest earnings are in the bottom third)
  does **not** help here: 10 days +0.88 → +0.82, 5 days +0.43 → +0.16. The 94 events it removes caught up
  *better* than the rest (+1.20 pt at 10 days). We keep it in the harness because it is a defensible risk rule: do
  not call a fundamentally weakening company a "temporary" laggard. But this table shows it is a judgement, not
  something the data proved. (The thirds are decided without look-ahead, against the previous quarter's cut-offs;
  see Report 4.)

<!-- structural -->
**Which evidence separates structural from temporary laggards?**

- **Sector strength** (Table 3b) separates them best. When the sector itself is falling, laggards do not catch
  up over 20 days (−0.65 pt); when the sector is rising they do (+1.18 pt). A stock left behind by a rising sector
  is more often temporary. A stock falling with a falling sector is just part of the sector's move.
- **Price recovery** (Table 3c) helps: +0.73 vs +0.36 at 10 days.
- **Macro risk** (Table 3d) matters. When XU100 is below its 200-day average or USD/TRY is spiking, catch-up
  mostly disappears (+0.21 vs +0.77). In stress periods investors sell the weaker names and do not rotate back.
- **Earnings tercile** (Table 3a) points the expected way but weakly. Across all no-shock laggards, those whose
  latest earnings are in the bottom third show no catch-up (+0.08 pt at 10 days, −0.05 at 20), while middle- and
  top-third laggards do (+0.65 / +0.48 at 10 days, +0.80 / +1.30 at 20). The t-values are small, and inside the
  harness's filtered set the bottom-third events caught up well (see the ladder), so this is a hypothesis, not a
  rule. "Statement public, quarter not ranked" are dates whose latest statement falls in the inflation-accounting
  switch (2022Q4 – 2024Q1), when no comparable previous-quarter cut-offs exist.

<!-- groups -->
Catch-up is clearest in Construction/Real Estate/Utilities (+1.09 pt), Banking (+0.92, but only 19 events) and
Retail & Consumer (+0.92). It is absent in Metals & Mining (−0.02) and Automotive (+0.02). Leaders in Chemicals &
Energy kept leading strongly (+2.15 pt). This group mixes a refiner (TUPRS), a fertiliser maker (GUBRF) and two
petrochemical firms (SASA, PETKM) whose prices follow different commodities, so a "sector median" there is a
weak benchmark. These per-sector numbers are each based on 19–207 events and were not corrected for 18 tests;
they describe the sample rather than prove differences.

<!-- false_positives -->
**False positives are the normal case, not the exception.**

- About half of no-shock laggards, including those that pass every harness filter, are still behind their
  peers 20 days later.
- The harness filters move the *median* outcome from about 0 to +0.47 pt, but they barely reduce the share of bad
  cases (28.9% vs 27.5% fell more than 5 pt further).

Table 6 shows what the worst cases have in common. In all of them **the peers rallied** (+18% to +51% in
20 days) while the stock stayed flat or fell:

| Case | Stock (20 days) | Peer median (20 days) |
|---|---|---|
| KCHOL, July 2023 | +26% | +51% |
| ISMEN, Sep 2023 | −3% | +30% |
| TCELL, Feb 2025 | −11% | +18% |

So the failure is usually "the sector ran away", not "the stock collapsed". The 3-member groups make this worse.
In Other Financials (KCHOL, ISMEN, TURSG), each stock's benchmark is the median of only two very different
companies: a holding, a broker and an insurer.

<!-- oos -->
This is the most encouraging table in the report. The full harness rule gives +0.88 pt in the first half and
+0.72 in the second half. Laggards that fail the harness filters drop from +1.00 to +0.02. The extra conditions are
what kept most of the effect in the later sample. Leaders stay slightly positive in both halves (momentum, not
reversal).

<!-- backtest -->
**The backtest is the reality check.** A small, stable 10-day edge at the event level did not become a
market-beating portfolio:

- **Whole period:** B_sektor returned 188% after costs, against 189% for XU100 and 342% for an equal-weight
  buy-and-hold of the 30 stocks.
- **First half:** it beat XU100 by 47 pt.
- **Second half:** it lagged XU100 by 24 pt.
- **Versus A_teknik:** B_sektor beats the same trend filter without sector ranking only slightly
  (+7 pt, Sharpe 1.29 vs 1.20, drawdown −24% vs −29%).
- **Costs:** doubling the cost to 0.30% one-way cuts the return from 188% to 134%. The strategy trades every week
  and is cost-sensitive.

The portfolio holds up to 10 stocks chosen mainly by the trend filter, not only catch-up events. Being in cash in
weak markets, and missing the 2022–2023 inflation rally in many names, costs more than the catch-up effect adds.
This is "prediction ≠ profitable strategy": a statistically significant event effect is not automatically a
strategy that beats a simple benchmark.

<!-- multiple -->
Table 1 contains 8 groups (2 settings × laggard/leader × shock/no shock). After Benjamini–Hochberg correction, only
the no-shock `lag10_5pct` laggard group has q < 0.05. The filter ladder, the structural splits (11 more
comparisons) and the 18 per-sector cells were not added to that correction. They are exploratory and are reported
with their day-clustered t so the reader can judge them. The one choice made before looking at outcomes is the
harness rule itself: its thresholds were set in Step 5. It is also the subset whose result held in the later
sample.

<!-- turkey -->
- **2022 – mid 2023: inflation rally and sector rotation.** With negative real rates, BIST rose strongly in
  nominal TRY and money rotated quickly between sectors. That favours catch-up: a stock left behind by its sector
  tends to be pulled along. The first-half results (other laggards +1.00 pt) reflect this.
- **Rallies where peers ran away.** Most of the worst false positives (KCHOL, ISMEN, TURSG, SASA in June–Sep
  2023; AKSEN, KCHOL and TCELL in Nov 2022) fall in strong, fast rallies. In June–July 2023 the market rose after the May
  2023 elections and the new economic team's return to orthodox policy, and leaders moved much faster than the
  rest. In such phases a laggard can stay behind even while it rises.
- **Second half (May 2024 – Nov 2026): tight policy and shocks.** Policy rates stayed high, and the March 2025
  political shock and the Sep–Oct 2026 fund-liquidation sell-off (see Report 1) hit the whole market. With the
  macro risk flag on, catch-up mostly disappeared (Table 3d), which matches the weaker second-half backtest.

The timing of the rallies is visible in our data. The political and policy events are from general knowledge
and from the sources checked for Report 1. The links between them and individual cases are our interpretation.

<!-- harness -->
This scenario is the basis of `POTENTIAL_CATCH_UP_CANDIDATE`. The harness:

1. **Checks the rule** on the analysis date: `lag10_5pct` laggard, no shock, sector median > 0, price
   recovering, and latest earnings not in the bottom third.
2. **Attaches the history:** the historical hit rate of no-shock laggards at 10 days, using only events whose
   10-day window had closed before the analysis date (about 52%, n ≈ 1,000+), and the B_sektor backtest up to that
   date.
3. **Demotes the label** with a shock or negative KAP news (`REJECT_SIGNAL`). If the stock lags but is not
   recovering or the sector is weak, it becomes `WATCH`.

Given this report, the label means "worth a closer look", never "expected to rise". A hit rate of about 52% is
close to a coin flip, and the explanation agents are required to say that.

<!-- limitations -->
- **Small peer groups.** Eight of the nine groups used have 3–4 members, so a sector median is the median of 2–3 other
  stocks. Healthcare (2 members) is excluded. Groups such as Other Financials and Chemicals & Energy mix different
  businesses.
- **Coarse fundamentals.** "Structural" is approximated by one earnings measure (year-over-year net-income
  third, decided against the previous quarter's cut-offs). Margins, debt and bank-specific ratios are not used here.
  Dates in the inflation-accounting switch (2022Q4 – 2024Q1) have no third.
- **Fixed thresholds.** The 5-pt / 10-day and 10-pt / 20-day thresholds and the 10-day dedupe gap were fixed in
  Step 5. Other values could give different numbers.
- **Strategy ≠ rule.** The backtest strategy differs from the harness rule: it is a weekly top-10 portfolio with a
  trend filter and does not use the earnings or sector-strength conditions.
- **Nominal TRY and survivorship.** Returns are nominal TRY (the gap vs peers is unaffected by inflation, the
  backtest is not), and the universe is the 2026 list.

<!-- conclusion -->
Sector-relative lag carries **some** information. Laggards without a shock, in a rising sector, that already
show signs of recovery, beat their peers by about 0.8 pt over the next 10 days (+0.88 in the first half, +0.72 in
the second). But half of them still fail, the effect is smaller than a single bad day, and a portfolio built on
the idea did not beat XU100 after costs. The right use is the one the harness makes: a laggard that passes the
rules is a **candidate for investigation**, shown with its historical hit rate, backtest and risks, and the
decision stays with a human.
