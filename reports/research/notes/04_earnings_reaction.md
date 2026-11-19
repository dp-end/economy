# Interpretation notes for Report 4 (edited by hand; the build script inserts each block by its key)

<!-- summary -->
**Short answer.** The market does react to quarterly results, and it reacts to *profitability*, not to sales.

- **Net-income ranking.** Ranked by year-over-year net-income change, top-third companies beat bottom-third
  companies by about **1.4 pt on the event day** and **1.7 pt in the five days before** disclosure (both q < 0.05).
  Part of the move happens before the KAP filing.
- **Which metrics matter.** Operating-income growth and net-margin change separate winners from losers on the event
  day at least as clearly as net income does, in both halves of the sample. Revenue growth, which in high-inflation
  Türkiye is mostly price increases, does not.
- **Drift.** After the first few days there is no reliable further drift for net income (days 1–20: +1.3 pt,
  p = 0.33, and the two halves disagree). The one exception is **ROE improvement**: +2.65 pt over days 1–20 in both
  halves (q = 0.003). It is the most interesting result for future work.
- **Not just the sector.** Against sector peers instead of XU100, the same pattern appears but is weaker
  (q ≈ 0.07–0.12). Part of the reaction is sector-wide, because peers report similar news in the same weeks.

<!-- main -->
**Reading Table 1, Table 2 and Figure 1.**

- **Before the event.** Top-third companies rise +1.6% above XU100 in the five days before the event day; bottom
  third is flat. The market partly anticipates good results: peers have often reported already, sector data is
  public, analysts publish estimates. We cannot tell anticipation from leakage with this data.
- **On the event day.** Bottom-third companies fall −1.34% vs XU100 and only 31% of them beat the index.
  Top-third companies are roughly flat (+0.11%).
- **Disclosure days are negative on average.** Every group, even companies with profit growth, has a negative
  average event-day return. The average across all events is about −0.6% vs the index, against a +0.03% baseline.
  Combined with the positive pre-event drift, this looks like "buy the rumour, sell the news". We report it as an
  observation, not a tested effect.
- **The spread is significant around the disclosure** (pre-event and event day) and **not after it**: days 1–5
  +0.46 pt, p = 0.49. Once the information is public it is priced within a few days. A rule that buys after reading
  the statement is mostly too late.

<!-- quarters -->
- **Across quarters.** The event-window spread is positive in 12 of 17 quarters and averages +1.76 pt. It is not
  stable.
- **Largest positive quarters.** 2022Q1 and Q2 (+6–7 pt), when nominal profit growth was extreme and dispersion
  between companies was large.
- **Negative quarters.**
  - 2022Q3 (−5.9 pt): results were published in Oct–Nov 2022, during a strong broad rally.
  - 2026Q3 (−2.6 pt): results were published during the Oct–Nov 2026 fund-liquidation sell-off (Report 1). In such
    weeks, market-wide flows dominate company news.
- **Inflation-accounting quarters** (orange, statements under TMS 29) show a steadier positive spread (+1.5 to
  +4.4 pt in most quarters). Real (inflation-adjusted) profit figures may be more informative than the nominal
  figures of 2022–2023. We have too few quarters to test this.

<!-- peers -->
Against sector peers the top-minus-bottom spread keeps its sign in every window (pre-event +1.30, event day +1.06,
days 0–4 +1.43 pt) but loses significance after correction.

Peers report in the same weeks, and sector-wide news (a bank-sector margin squeeze, steel prices) moves the whole
group. Some of the "stock reaction" is therefore a "sector reaction". The company-specific part is real but
smaller than the XU100-based numbers suggest.

<!-- metrics -->
**What the market reacts to is profitability, not growth in sales.**

- **Revenue YoY:** event-day spread +0.14 pt (p = 0.79), and the sign flips between halves. In 2022–2024, sales
  growth of 50–100% was mostly inflation, not volume, so it carried little news.
- **Operating income YoY:** +1.64 pt on the event day (q = 0.009), the same size in both halves (+1.62 / +1.66).
  Operating income is the closest line to EBITDA in our data; depreciation is not available.
- **Net margin change:** +1.69 pt on the event day (q = 0.005) and +2.36 pt over days 0–4 (q = 0.043), positive in
  both halves. Margins show whether a company passes inflation on to its prices.
- **ROE change:** a moderate event-day reaction, but a **+2.65 pt drift over days 1–20** (t = 5.2, q = 0.003,
  +2.87 / +2.43 in the two halves). This is the only post-disclosure drift that survives correction. We did not plan
  this test in advance and it is one of 22. It is a candidate finding to re-test on new quarters, not a rule.

<!-- direction -->
- **The worst case is not "profit → loss" but "profit down".** Companies whose profit fell (but stayed positive)
  lose −1.24% on the event day (only 29% beat the index) and −1.91% over days 0–4.
- **Loss → profit and profit → loss** are both negative on the day (about −1%) but recover over 20 days. With
  41–57 events each, these are noisy.
- **Small groups.** "Loss narrowed" (n = 17) and "loss widened" (n = 12) are too small to interpret. The +6.43% for
  "loss widened" is a few large rebounds.
- **Financials vs others.** The three financials react like the non-financials. Banks are ranked on net income and
  ROE only; revenue, operating-income and margin metrics exclude them.

<!-- tech -->
The spec asks to compare good fundamentals with weak technicals and vice versa. The pattern is consistent with
**surprise relative to expectations**, if the price trend is read as a proxy for expectations:

- **Strong trend, weak earnings** (bottom third, price above its 200-day average, n = 92): the largest event-day
  loss (−1.46%). Investors expected more.
- **Weak trend, weak earnings** (n = 58): only −0.38%. The bad news was already in the price.
- **Weak trend, strong earnings** (top third below its 200-day average, n = 21): the best event-window reaction
  (+1.87% over days 0–4), against +0.44% for top-third companies already in an uptrend.

These cells were not tested for significance and the "weak trend, strong earnings" group is small. It is an
interpretation to check on new data. It also shows why the harness never treats strong earnings alone as
sufficient.

<!-- spikes -->
Reporting season as a whole shows **no more** volume spikes or large abnormal moves than other weeks (ratios
0.87–1.11). The reaction is concentrated on each company's own event day, not spread across the season. This also
argues against widespread pre-announcement trading. The pre-event drift in Table 1 is more likely the market reading
public signals (peers' results, sector data) than leakage. We cannot prove either.

<!-- multiple -->
- **Six of 22 tests survive.** Two are net-income spreads (pre-event and event day), three are the
  operating-income and net-margin event-day/event-window spreads, and one is the ROE drift.
- **All six keep their sign in both halves** of the sample.
- **The 20-day net-income drift does not survive** (p = 0.33). Neither does any vs-peers spread, nor revenue.

The pattern of survivors is economically coherent: profitability measures, around the disclosure. That makes it
more believable than a single isolated p-value would be.

<!-- turkey -->
- **Nominal profits in 2022–2023.** With annual CPI above 50–80%, the median company's net income grew 75–200%
  year over year in nominal TRY. Ranking within each quarter is essential; raw growth rates would mostly measure
  inflation.
- **Inflation accounting (TMS 29) from 2023 year-end.**
  - KGK announced in November 2023 that companies applying TFRS must restate their 31 Dec 2023 statements for
    inflation.
  - SPK confirmed this for listed companies on 28 Dec 2023 and gave **10 extra weeks** for the first annual reports.
  - This is why 60–80% of 2023Q4–2024Q1 filings in our KAP data came after the usual deadline, and why we use real
    KAP timestamps.
  - Restated figures include a "monetary loss" line, so net income fell sharply for many companies. The 2024Q4
    median net-income change in our data is −73%.
- **Banks were treated differently.** BDDK decided that banks would not restate their 2023 statements. A December
  2025 BDDK decision says banks and other financial institutions will not apply inflation accounting to their
  2025 statements either. AKBNK's figures are therefore not on the same accounting basis as the industrial companies,
  another reason to rank them on net income and ROE only. We did not find a source for the 2024 bank decision.
- **2022Q4 missing.** Fintables lacks 2022Q4 for 24 of 30 stocks, so 2022Q4 and 2023Q4 (which needs 2022Q4 for the
  YoY change) cannot be ranked.

Verified by search: KGK and SPK TMS 29 announcements and BDDK bank decisions for 2023 and 2025
([KGK duyuru](https://www.kgk.gov.tr/Portalv2Uploads/files/Duyurular/v2/Diger/Y%C3%BCksek%20Enflasyonlu%20Sunum%20Para%20Birimine%20%C3%87evrim%20Duyuru.pdf),
[BloombergHT: SPK](https://www.bloomberght.com/spkdan-enflasyon-muhasebesi-aciklamasi-2344793),
[Sözcü: BDDK](https://www.sozcu.com.tr/bddk-dan-enflasyon-duzeltmesi-aciklamasi-p8411),
[Alomaliye: BDDK 2025](https://www.alomaliye.com/2025/12/23/enflasyon-muhasebesi-uygulamasi-banka-ve-finans-sirketlerinde-kaldirildi/)).
From our own data: the filing delays, the missing quarters and the median changes.

<!-- harness -->
The harness uses this scenario for `FUNDAMENTAL_IMPROVEMENT_CANDIDATE`. The rule requires:

- the latest **public** statement (event day ≤ analysis date) in the top net-income third;
- not a loss;
- no more than 63 trading days since the event day.

The attached history is the top-minus-bottom spread over days 0–4, computed only from events whose window had
closed before the analysis date. The evidence sentence states the quarter, the two net-income amounts, the
direction in words ("profit up", "loss narrowed", …) and the tercile.

Given this report, the label means "recent profitability news was strong". It does **not** predict further gains:

- most of the reaction happens before and on the event day;
- the label can be up to three months old.

ROE improvement, the one metric with post-disclosure drift, is not yet used by the harness. Adding it would need a
pre-registered test on new quarters first.

<!-- limitations -->
- **No expectations data.** There is no analyst consensus, so "surprise" is approximated by the change vs the
  same quarter last year and by the price trend.
- **No valuation or liquidity.** We have no share counts or market values (P/E, P/B) and no current
  assets/liabilities (liquidity ratios), so the spec's "valuation changes" and "liquidity" are not covered.
- **Operating income instead of EBITDA.** Depreciation is not in the downloaded statements.
- **Low power.** 17 quarters is a small number of independent observations. 2022Q4 and 2023Q4 are missing.
- **Comparability under TMS 29.** It is not documented whether Fintables restates prior-year figures, so
  year-over-year changes across the switch may mix accounting bases.
- **Same-day reaction partly missed.** For the 12% of filings published during trading hours, part of the reaction
  happens on the publication day, which counts as day −1 here.
- **Small, sensitive groups.** Only three financial companies. Returns are nominal TRY vs XU100 or peers.

<!-- conclusion -->
Quarterly fundamentals carry information that the market prices quickly:

- profitability changes (operating income, margins, net income) separate winners from losers around the
  disclosure;
- sales growth does not;
- most of the move happens within a few days, partly before the filing.

"Fundamental strength ≠ guaranteed price increase": strong results in a stock that had already run up reacted
little, and weak results in such a stock were punished most. The harness therefore uses fundamentals as **dated,
point-in-time evidence** next to the technical and sector picture, and only ROE improvement shows a post-disclosure
drift worth testing further.
