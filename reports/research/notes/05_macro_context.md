# Interpretation notes for Report 5 (edited by hand; the build script inserts each block by its key)

<!-- summary -->
**Short answer.** The context layers are essential for *interpreting* results but did not make better
*trading rules*.

- **Macro explains the numbers.** XU100 rose 7.5× in TRY over five years but only 1.4× in USD, and it lost to
  inflation in every full year except 2022. Most of the "returns" in Reports 1–4 are nominal, which is why every
  report measures against XU100 or peers.
- **Macro as a position-sizing rule hurt.** Halving positions when the macro risk flag is on cut the strategy's
  return from 241% to 176% and barely reduced its worst drawdown (−25% → −22%). In this sample, XU100 actually did
  *better* in the 20 days after the flag turned on (+5.4% vs +3.2%).
- **The lira matters for sectors.** When USD/TRY rises, metal and mining stocks (exporters, a gold miner) beat the
  index and banks lag it. Only the metals result survives correction.
- **News: one useful category.** Of the pre-registered KAP news categories, only "new contract / order"
  disclosures moved prices: +0.5 pt on the day (q = 0.002), mostly ASELS and EKGYO, with no follow-through.
  Negative categories showed no average effect. Adding news to the strategy (E) changed nothing.
- **Speeches: verified claims only.** Of 33 numerical claims in two central-bank speeches, 3 could be verified
  against our data and only those entered the evidence. Forecasts and targets stay as context: expert commentary
  ≠ verified fact.

<!-- macro -->
**The macro regime in three phases** (Figure 1, Table 2):

1. **Nov 2021 – mid 2023: negative real rates.** The policy rate was cut to 8.5% while yearly CPI peaked above 85%
   (late 2022). The real policy rate fell below −75 pt (lowest −76.5 pt in Nov 2022). Equities became an inflation hedge: XU100 rose +197% in 2022,
   the only year with a strongly positive real and USD return (+61% after CPI, +106% in USD).
2. **Jun 2023 – 2024: orthodox tightening.** The policy rate rose from 8.5% to 50% by March 2024. In 2023 the index
   still rose +36% in TRY but fell −14% in USD and −16% after inflation.
3. **Dec 2024 – Nov 2026: slow easing with shocks.**
   - Rates were cut from 50% to 36%.
   - The March 2025 political shock brought a temporary hike to 46% (April 2025, visible in our data).
   - The real policy rate stayed positive (+6–7 pt).
   - The index lagged both the lira and inflation in 2025 and 2026.

**Fed.** The Fed's path (0.25% → 5.5% → cuts to 3.75% → a hike to 4.00% in Sep 2026, from our hand-entered table)
mattered mainly through the dollar. It is far smaller in size than Türkiye's own rate swings.

**Consequence for the project.** A nominal TRY return of +200% in five years is roughly break-even in real
terms. The reports therefore judge signals by **excess return over XU100 or over peers**, never by raw TRY
return.

<!-- risk_flag -->
The macro risk flag was designed in Step 9 as a defensive rule: weak market trend or a sudden lira depreciation →
hold half positions. The data do not support it as a timing rule:

- **XU100 after the flag.** XU100 rose more in the 20 days after flagged days (+5.42%) than after normal days
  (+3.19%, p = 0.001). In Türkiye a lira shock or a sell-off has often been followed by a rebound: equities are
  bought as an inflation and currency hedge (Dec 2021, mid-2023), and sharp political sell-offs partly reversed
  (spring 2025).
- **Strategies.** Every D variant earns less than C. The interest-rate variant earns the least (111%), and the
  drawdown improves by only 3 pt. Adding the news filter (E) gives 171% vs 176% for D.
- **No help in the second half.** In the tighter second half, all of them lag XU100 (+2–11% vs +33%).

The macro layer stays in the harness as **context and as a gate warning**, not as an automatic sizing rule. This
was the main negative result of Step 9, and Table 3 shows it was not just bad luck in the backtest.

<!-- fx -->
- **Metals & Mining** (EREGL, KRDMD, TRALT): beta +0.12 (t = 3.2, q = 0.014). Steel exporters and a gold miner
  earn in dollars, so a weaker lira helps them relative to the market.
- **Banking:** beta −0.13 (p = 0.025, q = 0.12 after correction). Lira weakness usually comes with expectations of
  tighter policy and higher funding costs, which hurts bank margins.
- **Retail & Consumer** is mildly negative: import costs and squeezed consumers.
- **Other groups:** close to zero.

These are average daily co-movements, not forecasts. The harness can use them to explain *why* a stock moved on
a lira-shock day, not to predict the next move.

<!-- news -->
**Sources.**

- **KAP:** the official disclosure platform, the only news source with a full five-year history
  (12,905 disclosures).
- **RSS:** the feeds (AA, BloombergHT, TRT Haber) give only the latest headlines and were archived from
  13 Nov 2026, so they are context for the present, not for backtests.
- **Tagging:** company, sector and macro tags come from keyword rules (Table 7, Table 8). No language model labels
  news.

**Price reaction (Table 9).** The rules were fixed on 2026-11-18, before any outcome was looked at.

- **New contract / order** (283 events, 14 stocks): +0.54% on the event day vs a +0.04% baseline (q = 0.002), and
  some anticipation before (+0.71% in the five days before). There is no further move after day 0 (days 1–5:
  +0.05%), so the information is priced at once.
- **Who drives it:** ASELS (139 events, defence contracts) and EKGYO (113, real-estate tenders) produce most of
  the events, so this is close to a two-company finding.
- **Negative categories** (lawsuits/penalties, executive departures) show **no average negative effect**. Many
  such filings are routine (small fines, planned retirements), and the keyword rules cannot tell a serious event
  from a routine one. "Liquidation" mostly matches the liquidation of small subsidiaries, which is often good news.
- **Small categories:** buy-back starts (1 event) and delayed statements (4) are too rare to interpret.

**How the harness uses news.** It treats negative KAP news as a reason to **investigate** (TURSG, Step 14), not as
a sell signal. Table 9 supports that caution: on average such news is not bad for the price, but in a single case
it can be.

<!-- speech -->
**Sources.** No instructor-approved list arrived, so we used only official central-bank sources:

- **Fed, FOMC press conference, 28 Oct 2026.** The audio was taken from the Fed's own video player and
  transcribed with Whisper (small model). It matches the Fed's official transcript at 96.6%.
- **TCMB, Inflation Report briefing, 12 Nov 2026.** We used the official published text. The YouTube video
  required a bot sign-in, which we did not bypass, the same principle as for KAP.

**Claims (Table 12).** Every number in the speeches was extracted and checked against our own data tools:

- **3 verified:** the Fed range 3.75–4.00%, Türkiye's yearly CPI 29.8% and the TCMB policy rate 36%.
- **1 consistent with history:** a past rate path.
- **8 forecasts or targets** (for example the 2% inflation target): they cannot be true or false today.
- **21 with no matching tool** (growth, wages, other variables).

Only the 3 verified claims are allowed into the committed evidence (Table 13). The rest can be shown as context,
labelled as statements by the speaker.

**Stance.** Stance is a keyword count (hawkish / dovish / uncertainty words per sentence). The Fed conference
reads clearly hawkish (28 hawkish sentences, 0 dovish). The TCMB briefing is hawkish in tone but has the most
uncertainty language (31 sentences), while TCMB had just *cut* rates. This shows why tone must not be confused with
action: the data tools, not the speech, tell the harness what the policy rate is.

<!-- harness -->
Every analysis builds an **evidence bundle**, and each layer enters it in a controlled way:

- **Macro:** one committed sentence with the policy rate, the latest *published* CPI (with its reference month)
  and the macro risk flag. The quality gate warns if any macro series is missing on the analysis date.
- **News:** KAP disclosures from the last 10 trading days, each with date, category, title and link. A negative
  one blocks candidate labels (→ `INVESTIGATE`, or `REJECT_SIGNAL` if it hits a candidate). Positive ones are shown
  but never create a candidate on their own. The gate warns when a stock has fewer than 3 KAP disclosures in
  180 days (sparse news coverage).
- **Speech:** features of the latest official speech within 30 days: stance counts, topics and **verified**
  claims only. The gate warns about transcript quality problems.

The explanation agents (Step 13) may only use numbers present in this bundle. The numerical validator rejects
anything else. A speech cannot inject a number, and an LLM cannot "remember" a macro value.

<!-- turkey -->
This report *is* largely the Türkiye context for the project. The dated events used across all reports:

- **20 Dec 2021: FX-protected deposits.** A currency-crisis response, shortly after our sample starts. USD/TRY
  rose 37% in the last two months of 2021 alone (Table 2).
- **2022: negative real rates.** Inflation peaked above 85% in late 2022. Equities became the inflation hedge.
- **6 Feb 2023: earthquakes.** Borsa İstanbul was closed 8–14 Feb 2023 (excluded in all event studies).
- **Jun 2023 – Mar 2024: tightening** from 8.5% to 50% after the May 2023 elections.
- **Dec 2023: inflation accounting (TMS 29)** for listed non-financial companies (Report 4).
- **19 Mar 2025: political shock.** Two circuit-breaker halts, a ~16% weekly loss and an emergency hike to 46%
  (April 2025).
- **17 Sep 2026: fund liquidations.** SPK ordered 131 funds liquidated, followed by a ~25% two-month fall to
  late October 2026. Our data ends on 6 Nov 2026, inside this sell-off.

Verified by search in Reports 1 and 4: the March 2025 and Sep–Oct 2026 events and the TMS 29 decisions. From our
own data: all rate, CPI and FX values and the 2025 hike. From general knowledge: the 2021 scheme and the
earthquake closure.

<!-- limitations -->
- **Release dates are rules, not the real calendar.** CPI availability follows a rule (3rd of the next month,
  business-day adjusted), not the actual TÜİK calendar. Data revisions are not modelled; the latest vintage is used.
- **FX rate timing.** The TCMB indicative FX rate is published in the afternoon. We assume it is known at the BIST
  close.
- **Fed table.** The Fed rate table is entered by hand from the Fed's website.
- **News history.** RSS has no history, so all news statistics rely on KAP. Keyword categories are coarse and
  cannot judge severity.
- **Speeches.** Only two speeches, chosen by us because no approved list was provided. There is no speech-day
  event study, so whether speeches move BIST is not tested (an optional extension). Stance is a keyword count.
- **Model choices.** FX betas are simple daily regressions on nominal returns. The macro flag thresholds were fixed
  in Step 9 and not optimised, which keeps the negative result honest but means other thresholds might differ.

<!-- conclusion -->
- **What the context layers gave:** they explained *why* raw TRY numbers look the way they do, which sectors move
  with the lira, which news types the market prices, and which statements in a speech are facts.
- **What they did not give:** a better trading rule. The macro filter lowered returns, the news filter changed
  nothing, and the one priced news type (new contracts) is priced on the day.
- **How the project uses them:** as **dated, verified context** in the evidence bundle and as **warnings** in the
  quality gate. The human reviewer sees them next to the numbers, and no layer is allowed to turn an opinion or a
  headline into a number.
