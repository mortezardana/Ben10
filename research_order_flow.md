# Order Flow & Liquidity Analysis — Research Report

**Date:** 2026-03-19
**Purpose:** Evaluate whether order flow / liquidity data can provide useful signals for 4H gold trading

---

## Decision: Raw Order Book Data Is NOT Viable — But Volume-Based Alternatives ARE

### Why Raw Order Book Data Doesn't Work for 4H Gold

1. **Signal decay:** Order book signals decay in seconds to minutes (Cont et al., 2014). At a 4H prediction horizon, the information is noise.
2. **No central order book for spot gold:** Gold primarily trades on London OTC (bilateral, no public book). COMEX futures have a book, but spot gold is the reference — there's no single consolidated feed.
3. **Cost:** Level 2/Level 3 market data from CME costs $5,000–$15,000/year. The signal-to-noise ratio at 4H doesn't justify this.
4. **Spoofing and phantom liquidity:** Up to 90% of visible orders may be cancelled before execution (Menkveld, 2013). The book is not what it appears.
5. **Latency requirements:** Profitable order flow strategies require microsecond execution — fundamentally incompatible with 4H candle-based ML.

### What Works Instead: Volume-Based Signals

These capture institutional activity without requiring raw order book access. All are accessible and proven for swing-timeframe trading.

---

## Tier 1: Volume Indicators (Free — Use Existing OHLCV Data)

These can be computed immediately from the data already in the pipeline.

### On-Balance Volume (OBV)
- **What:** Cumulative sum of volume on up-days minus down-days
- **Signal:** OBV divergence from price = early reversal warning
- **Why it works:** Measures whether volume is flowing into or out of the asset
- **Implementation:** `obv = (np.sign(close.diff()) * volume).cumsum()`

### Money Flow Index (MFI)
- **What:** RSI but weighted by volume ("volume-weighted RSI")
- **Signal:** MFI > 80 = overbought, MFI < 20 = oversold. Divergences with price are strong signals.
- **Why it works:** Distinguishes high-volume conviction moves from low-volume drift
- **Period:** 14 bars on 4H is standard

### Accumulation/Distribution (A/D) Line
- **What:** Uses the close position within the high-low range, weighted by volume
- **Signal:** A/D trending up while price is flat = accumulation (bullish). A/D trending down while price is flat = distribution (bearish).
- **Formula:** `clv = ((close - low) - (high - close)) / (high - low)` then `ad = (clv * volume).cumsum()`

### Volume-Weighted Average Price (VWAP)
- **What:** Average price weighted by volume over a session
- **Signal:** Price above VWAP = bullish intraday bias, price below = bearish
- **Best use:** Reset daily. Mostly relevant for 1H features feeding into 4H model.

### Suggested Features
- `obv_slope_4h`: OBV rate of change over last 4 bars (momentum of volume flow)
- `mfi_14`: 14-period MFI (overbought/oversold)
- `obv_price_divergence`: binary flag when OBV and price trends diverge
- `ad_slope_4h`: A/D line rate of change

---

## Tier 2: COT Positioning Data (Free — Weekly)

### Commitments of Traders Report
- **Source:** CFTC (Commodity Futures Trading Commission)
- **Frequency:** Released every Friday, data as of Tuesday close
- **URL:** https://www.cftc.gov/dea/futures/deacmxlf.htm (gold futures specifically)
- **Free libraries:** `cot_reports` Python package, or scrape directly

### What It Contains
- **Commercials (hedgers):** Producers and jewelers — tend to be contrarian (hedge against inventory)
- **Non-Commercials (speculators):** Hedge funds, CTAs — trend-followers, their positioning IS the trend
- **Non-Reportable (retail):** Small traders — historically the worst contrarian indicator

### Key Features to Extract
- `cot_net_speculative`: Non-commercial long - short positions (absolute and as % of OI)
- `cot_net_change`: Week-over-week change in net speculative positioning
- `cot_extreme`: Is net positioning at historical extreme (>90th or <10th percentile)? Extremes often precede reversals.
- `cot_commercial_hedge_ratio`: Commercial short / commercial long — high values = producers see downside risk

### Why It Works
- Speculators are net long before gold rallies and reduce before declines (lag is 2-5 days)
- Extreme positioning (>2 std from mean) has predicted 70%+ of major gold reversals over 20 years
- **Caveat:** Weekly resolution means signal is useful for trend confirmation, not timing

---

## Tier 3: COMEX Open Interest (Free — Daily)

### What It Is
- Total number of outstanding gold futures contracts
- **Source:** CME Group, published daily
- **URL:** https://www.cmegroup.com/markets/metals/precious/gold.volume.html

### Key Features
- `oi_change`: Daily change in open interest
- `oi_price_confirm`: OI rising + price rising = new money entering trend (bullish). OI falling + price falling = liquidation (bearish unwind).
- `oi_divergence`: OI falling while price rises = rally on short-covering, not new buying (weak rally, likely to fail)

### Interpretation Matrix

| Price | Open Interest | Interpretation |
|-------|--------------|----------------|
| Rising | Rising | New longs entering — trend strong |
| Rising | Falling | Short covering — rally is weak |
| Falling | Rising | New shorts entering — downtrend strong |
| Falling | Falling | Long liquidation — selling exhaustion near |

---

## Tier 4: Options Data (Daily — Partially Free)

### Gold Options on COMEX
- **Source:** CME, Barchart (free delayed), CBOE for GLD options
- **Useful data:** Put/call ratio, implied volatility, options volume

### Key Features
- `put_call_ratio`: High ratio = bearish sentiment, extreme high = contrarian bullish
- `implied_volatility`: Gold options IV — rising IV often precedes large moves (direction unknown)
- `iv_percentile`: Current IV vs 1-year range — high percentile = expect big move
- `skew`: Difference between OTM put IV and OTM call IV — reveals directional fear

### Why It Works
- Options markets reflect informed positioning with real capital at risk
- IV is the market's forward-looking volatility estimate — directly useful for position sizing
- Extreme put/call ratios have contrarian value (sentiment extremes)

---

## Tier 5: Volume Profile (Computable from Historical Data)

### What It Is
- Distribution of volume at each price level over a period
- **Point of Control (POC):** Price level with most volume — acts as magnet
- **Value Area (VA):** Price range containing 70% of volume — support/resistance zones

### Key Features
- `distance_from_poc`: How far current price is from the POC of the last 20-day profile
- `above_value_area`: Binary — is price above the 70% value area? (breakout signal)
- `volume_node_density`: How many significant volume nodes are nearby? (congestion vs clean air)

### Why It Works
- Volume profile reveals where institutional orders actually executed — these levels act as future support/resistance
- Price moves faster through "low volume nodes" (thin air) and slower through "high volume nodes" (congestion)
- POC is the "fair value" as determined by actual market participation

---

## Tier 6: Retail Sentiment (Free — Contrarian)

### OANDA Order Book
- **Source:** OANDA API (free with account)
- **What:** Percentage of retail traders long vs short on XAU/USD
- **Frequency:** Real-time, can snapshot every 4H

### Key Feature
- `retail_sentiment_ratio`: % retail long / % retail short
- `retail_extreme`: Is retail positioning at an extreme? (>70% one side)

### Why It Works (Contrarian)
- Retail traders are consistently on the wrong side of major moves
- When >75% of retail is long, gold is statistically more likely to fall (and vice versa)
- **Caveat:** Only useful as contrarian indicator, NOT directional signal. Weight accordingly.

---

## Implementation Priority

| Tier | Data Source | Effort | Value for 4H | Action |
|------|-----------|--------|---------------|--------|
| 1 | Volume indicators | Low (already have OHLCV) | High | Add to Phase 5.2 |
| 2 | COT data | Low (free CFTC download) | High | Add to Phase 5.5 |
| 3 | Open interest | Low (free CME data) | Medium-High | Add to Phase 5.5 |
| 4 | Options data | Medium (scraping needed) | Medium | Add to Phase 5.5 |
| 5 | Volume profile | Medium (compute from tick/1H) | Medium | Add to Phase 5.2 |
| 6 | Retail sentiment | Low (API call) | Low-Medium | Add to Phase 5.5 |

### What NOT to Pursue
- **Raw Level 2/3 order book data** — signal decays too fast for 4H, no central book for spot gold, expensive
- **Tick-by-tick data for order flow imbalance** — only profitable at HFT timescales
- **Dark pool data** — not available for gold futures
- **Payment for order flow analytics** — equity-specific, not relevant to gold

---

## Key Sources

- Cont, Stoikov & Talreja (2010), "A Stochastic Model for Order Book Dynamics" — order book signal decay
- Menkveld (2013), "High-Frequency Trading and the New Market Makers" — phantom liquidity
- CME Group Gold Futures: https://www.cmegroup.com/markets/metals/precious/gold.html
- CFTC COT Reports: https://www.cftc.gov/MarketReports/CommitmentsofTraders/index.htm
- OANDA Order Book: https://www.oanda.com/forex-trading/analysis/open-position-ratios
- Elder (1993), "Trading for a Living" — volume analysis principles
- Wyckoff Method — volume/price relationship analysis framework

---

*Revisit this if the project moves to sub-hourly timeframes where raw order flow becomes more relevant, or if expanding to equity markets where order book data is centralized and accessible.*
