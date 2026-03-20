I# Algorithmic Trading — Deep Research Report

**Date:** 2026-03-19
**Purpose:** Inform the evolution of the Ben10 gold trading pipeline

---

## 1. Trading Strategies — What Actually Works

### Proven Strategy Classes

**Mean Reversion** — the most empirically validated class. Prices that deviate from a historical mean tend to revert. Works best at shorter timeframes (intraday to days) and in range-bound markets. Gold was range-bound from 2013–2018 — mean reversion would have thrived there.

**Momentum / Trend Following** — decades of evidence (AQR, Asness et al. 2013). Works across all asset classes and timeframes. CTA funds trading medium-to-long term momentum have shown persistent returns, though the edge has compressed as more capital chases it. Gold has historically shown strong momentum during macro regime shifts — the 2020–2026 super-cycle is a prime example.

**Statistical Arbitrage / Pairs Trading** — exploits temporary price discrepancies between correlated instruments. Renaissance Technologies and DE Shaw built empires on variations of this. For gold, pairs between XAU/USD and DXY, silver, or treasury yields are directly applicable.

**Market Making** — HFT strategy requiring sub-millisecond execution. Not relevant for this project but worth noting: this is where pure RL has shown the most production success.

### Frequency Tiers

| Aspect | HFT (microseconds) | Medium Frequency (minutes–hours) | Low Frequency (days–months) |
|--------|-------------------|----------------------------------|----------------------------|
| Infrastructure cost | Very high (co-location, FPGAs) | Moderate (good server, API) | Low (laptop is fine) |
| Edge source | Speed, microstructure | Statistical patterns, ML signals | Macro factors, fundamentals |
| Alpha decay | Days to weeks | Weeks to months | Months to years |
| Accessibility | Institutional only | Accessible to skilled individuals | Most accessible |

**Ben10 operates at medium frequency (4H candles).** This is the sweet spot for ML — enough data to train, slow enough that execution quality isn't the primary determinant of success, fast enough to accumulate statistically meaningful trade counts.

### What Only Works in Papers

- Over 90% of academic strategies fail when implemented with real capital
- Transaction costs erode theoretical alpha (especially high-turnover strategies)
- Survivorship bias inflates historical returns by 1–4% annually
- Many papers implicitly use look-ahead information in feature construction

---

## 2. Machine Learning in Trading — What's Proven

### Gradient Boosting Is King

XGBoost, LightGBM, and CatBoost dominate production trading for concrete reasons:

1. **Handle tabular data natively** — financial features are inherently tabular
2. **Built-in feature importance** — critical for understanding and trusting predictions
3. **Robust to overfitting** with proper regularization
4. **No feature scaling required** — tree-based models are invariant to monotonic transformations
5. **Handle missing data** natively
6. **Fast training** — LightGBM with GOSS processes massive datasets in seconds

| Model | Strength | Best For |
|-------|----------|----------|
| XGBoost | Highest accuracy, most mature, best interpretability | Primary model — solid choice for Ben10 |
| LightGBM | Fastest training, best for large datasets | Direct comparison to XGBoost — add this |
| CatBoost | Best with categorical features, lowest overfitting risk | Good if adding categorical features (day-of-week, session, regime labels) |

**Key insight: Simple models with strong features consistently outperform complex models with raw data.**

### Deep Learning — What Actually Works

**LSTMs:** Most validated DL approach for financial time series. 53.3% RMSE reduction vs ARIMA across 25 years of data. Performance varies by asset and market condition. Ben10 already has this.

**CNN-LSTM hybrids:** CNNs extract local patterns, LSTMs capture temporal dependencies. 2025 research continues to show these outperform standalone models. Ben10 already has this.

**Transformers:** State of the art but nuanced. ICML 2025 finding: "simpler transformers consistently outperform their more complex counterparts." The attention mechanism captures long-range dependencies well, but financial time series have low signal-to-noise ratios that can confuse self-attention.

**TCN (Temporal Convolutional Networks):** Competitive with LSTMs while being faster to train. Dilated causal convolutions handle long sequences efficiently. Often overlooked but strong. Ben10 already has this.

### Transformer Architectures Worth Implementing

**Temporal Fusion Transformer (TFT)** — most promising for finance:
- Interpretable attention weights (see which features/timepoints matter)
- Handles known future inputs (calendar events) and unknown future inputs
- Variable selection networks built in
- Caveat: "direct use often struggles with market non-stationarity"
- **Recommendation: This should be the next architecture added to Ben10.**

**PatchTST** — segments time series into patches before attention. Up to 21% lower MSE vs alternatives. Strong general model but less domain-specific than TFT.

**Informer** — efficient for long sequences with ProbSparse attention (O(n log n) vs O(n²)). Good for multi-horizon forecasting.

### Reinforcement Learning — Honest Assessment

**Mixed results for position-taking strategies.**

- Best success: market making (continuous action spaces, clear rewards)
- For direction-based trading like Ben10: inconsistent. One study showed RL achieving Sharpe 1.23 vs buy-and-hold's 1.46 — underperforming
- Ensemble RL agents show better results than individual agents
- Key challenges: non-stationarity, low signal-to-noise, difficulty defining reward functions
- "Implementation quality and domain knowledge often outweigh algorithmic complexity"

**Recommendation: RL should remain low priority (Phase 6 of roadmap). Fix the foundation and get gradient boosting + proper validation working first.**

### Foundation Models for Time Series

The fastest-moving frontier (2024–2026):

| Model | Developer | Key Feature | Status |
|-------|-----------|-------------|--------|
| TimeGPT | Nixtla | 100B+ data points | Closed-source, API-based |
| TimesFM | Google | Decoder-based, patch-based | Open-source, strong zero-shot |
| Chronos | Amazon | T5-based, tokenized time series | Open-source, benchmark leader |
| Chronos-2 | Amazon (Oct 2025) | Multivariate, covariates | 300+ forecasts/sec on single GPU |
| Moirai | Salesforce | Mixture distributions | Open-source, 36M training series |
| Moirai 2.0 | Salesforce | Decoder-only, quantile forecasting | Latest, multi-token prediction |

**Practical reality:** These excel at forecasting but aren't designed for trading decisions. They predict "what will the price be" not "should I trade." Use them as feature generators (feeding forecasts into XGBoost/ensemble) rather than standalone systems.

### LLM-Based Approaches

Most practical application for Ben10: **sentiment analysis** of gold-related news, FOMC statements, and geopolitical events, feeding sentiment scores as features into existing models. 87% forecast accuracy reported for social media sentiment in some studies.

LLM trading agents: "they faithfully follow directions regardless of profit implications" — they don't inherently optimize for profit.

---

## 3. Known Pitfalls and Failures

### Overfitting

The single most dangerous problem. Knight Capital lost $440M in 45 minutes from a deployed overfitted algorithm.

Signs of overfitting:
- Dramatically better backtest results than out-of-sample
- Performance degrades sharply with small parameter changes
- Many parameters relative to number of trades
- Strategy only works on one specific asset/period

**Ben10 risk: With 161+ indicators as features, the project is at extreme overfitting risk.** Feature selection is critical — start with 10–20 features maximum and justify each with economic reasoning.

### Look-Ahead Bias

"Often the main reason why trading strategies underperform their backtests significantly." Common sources:
- Computing indicators on the full dataset before splitting
- Features that include information not available at prediction time
- Normalizing with statistics from the full dataset

**Ben10 has this problem right now** — normalization leakage is the #1 fix.

### Transaction Cost Erosion

A strategy showing 10% annual returns before costs might show -2% after realistic costs. For gold on 4H:
- Spread: 0.3–0.5 pips institutional, 1–3 pips retail
- On 4H bars with moderate trading frequency, costs are manageable but must be modeled

### Regime Changes and Non-Stationarity

Markets are non-stationary — statistical properties change over time. A model trained on the 2010–2019 low-volatility era may fail in 2020's COVID volatility. Gold specifically:
- 2004–2011: Bull run
- 2013–2018: Range-bound
- 2020–2026: Breakout super-cycle

One model doesn't fit all regimes.

### Alpha Decay

| Strategy Type | Typical Decay |
|---------------|--------------|
| HFT | Days to weeks |
| Momentum | 3–6 months (60% decay within ~10 months) |
| Swing/position | 6–18 months |
| Macro/fundamental | 1–3 years |

Causes: overfitting, factor crowding (too many traders exploit the same signal), structural market changes.

### The Backtest-to-Live Gap

Over 90% of academic strategies fail with real capital. Causes:
- Market impact (your orders move the price)
- Execution uncertainty (slippage, partial fills, latency)
- Emotional interference (overriding signals)
- Data quality differences between backtest and live

### Lopez de Prado's "10 Reasons Most ML Funds Fail"

1. Using standard cross-validation instead of purged/embargo'd CV
2. Treating financial problems as typical supervised learning
3. Ignoring the multiple testing problem
4. Not accounting for non-IID data
5. Confusing prediction accuracy with profitability
6. Overcomplicating models when simple ones suffice
7. Not having a theory for why a pattern should exist

---

## 4. Feature Engineering for Financial ML

### What Features Actually Matter

**Core price-derived features (highest signal):**
- Returns at multiple horizons (1-bar, 4-bar, 12-bar)
- Volatility (realized vol, ATR, Garman-Klass)
- Volume and volume ratios
- Trend strength (ADX)
- RSI
- Moving average relationships (price vs SMA, SMA crossovers)

**The 161+ indicator problem:** Most are redundant or noise. Research consistently shows a curated set of 15–30 features, each with economic rationale, outperforms hundreds of indicators. Use SHAP values from XGBoost to identify which features carry predictive power.

### Technical Indicators vs Raw Price Features

Both have value, but raw features (OHLCV returns, volatility) often outperform complex indicators because they're more stable over time, less prone to parameter overfitting, and models can learn indicator-equivalent representations from raw data.

**Best practice:** Mix of raw features + small number of well-chosen indicators.

### Lopez de Prado's Key Contributions

**Triple Barrier Method:** Replace simple binary labels with path-dependent labels:
- Upper barrier (take profit): hit first → label +1
- Lower barrier (stop loss): hit first → label -1
- Time barrier (expiration): hit first → label 0

This produces more realistic labels that account for path dependency. **Highly recommended for Ben10** — replace current binary classification target.

**Meta-Labeling:** Two-stage approach:
1. Primary model generates buy/sell signals
2. Secondary ML model decides whether to act on the signal (trade or pass)

Dramatically reduces false positives. For Ben10's ensemble phase: best model generates signals, meta-learner filters them.

**Fractional Differentiation:** Standard differencing (returns) makes series stationary but destroys memory. Fractional differentiation (d ≈ 0.2) achieves stationarity while retaining >90% correlation with the original series. Library: `fracdiff` on PyPI.

**Combinatorial Purged Cross-Validation (CPCV):** Generates multiple chronology-respecting train/test partitions with purging (removes near-boundary observations) and embargo (gap after purging). Shows "marked superiority in mitigating overfitting risks."

### Alternative Data for Gold (Ranked by Proven Value)

1. **FOMC statement sentiment** — rate expectations directly move gold
2. **Geopolitical risk indices** — gold is the classic safe-haven
3. **COT reports** — institutional positioning in gold futures
4. **Gold ETF flows (GLD, IAU)** — large inflows/outflows move price
5. **Economic Policy Uncertainty Index (EPU)** — macro uncertainty drives gold demand
6. **DXY, Treasury yields, VIX** — macro drivers (already on roadmap)

---

## 5. Validation and Testing

### Walk-Forward Analysis

Industry standard. Divide data chronologically: optimize on in-sample, test unchanged on out-of-sample, roll forward. Walk-Forward Efficiency = Out-of-Sample Return / In-Sample Return. Values >70% indicate strong transferability. **This is Phase 3 of the roadmap — critical.**

### Combinatorial Purged Cross-Validation (CPCV)

Lopez de Prado's improvement:
- Multiple chronology-respecting train/test partitions
- Purging: removes observations near train/test boundary
- Embargo: adds gap period after purging
- Produces distribution of performance metrics
- Lower Probability of Backtest Overfitting (PBO)

### Deflated Sharpe Ratio (DSR)

Corrects for:
1. Selection bias under multiple testing
2. Non-normally distributed returns

If you test 100 parameter combinations, the required Sharpe for 95% confidence may be 3.0+. Always report how many strategies were tested.

### Monte Carlo Simulation

Run 10,000+ simulations with:
- Reshuffled trade sequences (tests ordering dependence)
- Randomized parameters (tests parameter sensitivity)
- Skipped trades (simulates missed signals)
- Resampled returns (generates confidence intervals)

Robust strategies show consistent performance across all variations.

---

## 6. Risk Management

### Position Sizing

**Kelly Criterion:** Kelly% = W − [(1−W) / R] where W = win rate, R = win/loss ratio.

**Critical: Full Kelly is almost never used** — extreme drawdowns.
- Half Kelly: ~75% of optimal growth, ~50% less drawdown
- Quarter Kelly: most common among professionals
- Combine with VaR/CVaR constraints

### Volatility Targeting

Scale positions inversely to volatility so each has equal risk contribution. Use ATR for intra-day scaling. Target specific portfolio volatility (e.g., 10% annualized). **Directly applicable to gold** — volatility varies enormously across regimes.

### Key Metrics

| Metric | What It Measures | Target |
|--------|-----------------|--------|
| VaR (95%) | Max loss in 95% of periods | Strategy-dependent |
| CVaR | Average loss in worst 5% | More conservative than VaR |
| Max Drawdown | Largest peak-to-trough decline | < 20% |
| Sharpe Ratio | Risk-adjusted returns | > 1.0 good, > 2.0 excellent |
| Sortino Ratio | Downside risk-adjusted returns | Better than Sharpe for asymmetric returns |
| Calmar Ratio | Return / Max Drawdown | Higher is better |

### Portfolio Construction

- **Risk Parity:** Equal risk contribution from each strategy/asset
- **Hierarchical Risk Parity (HRP):** Clustering-based, reduces estimation error
- **Maximum Diversification:** Maximize ratio of individual to portfolio volatility

---

## 7. Gold-Specific Analysis

### What Drives Gold Prices (Ranked by Impact)

1. **Real interest rates** — strongest inverse correlation. When real yields fall, gold rises.
2. **US Dollar strength (DXY)** — strong inverse correlation. Dollar weakness = gold strength.
3. **Central bank buying** — structural driver since 2022. China, India, Turkey accumulating aggressively. Biggest structural shift in decades.
4. **Geopolitical uncertainty** — safe-haven demand.
5. **Inflation expectations** — gold as inflation hedge, mediated through real rates.
6. **Monetary policy** — Fed rate decisions directly affect gold through rates channel.
7. **ETF flows** — large gold ETF inflows/outflows move prices.

### 2024–2026 Gold Context

Gold is in a historic super-cycle:
- Climbed 55% in 2025, surpassing $4,000/oz
- Surged above $5,000/oz, hit $5,595 intraday on Jan 29, 2026
- J.P. Morgan forecasts $5,055/oz average by Q4 2026
- Broke from a rising wedge forming since the 1980s

### Gold-Specific ML Results

- LSTM-MLP: MAE of $0.21 on daily gold closing prices
- CNN-LSTM: effective feature extraction for gold
- Hybrid econometrics + ML + DL frameworks show best results
- GA-optimized MLP: R² of 0.98 on 11-year dataset

### Unique Characteristics

- Trades 23 hours/day (nearly continuous)
- Liquidity varies by session (thinnest during Asian session)
- Heavily influenced by macro events — FOMC, NFP, CPI releases cause sharp moves
- Persistent momentum during macro regime shifts
- Mean reversion works during range-bound periods

---

## 8. Production Landscape — What Top Firms Actually Use

### Top Quant Fund Performance (2024)

- Renaissance Medallion: 30% return
- Renaissance Institutional Funds: 22.7% and 15.6%
- Two Sigma Spectrum: 10.9%
- Two Sigma Absolute Return Enhanced: 14.3%

### What They Actually Use

- Statistical arbitrage with alternative data
- Multi-factor models (value, momentum, quality, volatility)
- ML for alpha generation (primarily gradient boosting + feature engineering)
- NLP on news, earnings calls, regulatory filings
- Massive engineering investment in data pipelines
- Continuous retraining and monitoring
- Python for research, C++/Rust for production execution

### Tech Stack Reality

The best performing approaches in practice:
1. Gradient boosting on well-engineered features
2. Ensemble methods (meta-labeling and stacking)
3. Alternative data (sentiment, macro)
4. Walk-forward validation with proper purging
5. Position sizing and risk management equal in importance to signal quality
6. Continuous monitoring and retraining to combat alpha decay

---

## 9. Essential Resources

### Books (Ranked by Practical Value)

1. **"Advances in Financial Machine Learning"** — Marcos Lopez de Prado. THE book. Triple barrier, meta-labeling, fractional differentiation, CPCV, structural breaks. Required reading.
2. **"Machine Learning for Algorithmic Trading" (2nd ed.)** — Stefan Jansen. Most comprehensive code-focused book.
3. **"Quantitative Trading"** — Ernest Chan (2025 edition). Best introduction for building your own system.
4. **"Python for Algorithmic Trading" (2nd ed.)** — Yves Hilpisch. Advanced coding and execution models.
5. **"Machine Learning for Factor Investing"** — Coqueret and Guida. Bridges ML and traditional quant finance.

### Key Papers

- Sharpe (1994): "The Sharpe Ratio"
- Jegadeesh & Titman (1993): "Returns to Buying Winners and Selling Losers"
- Asness et al. (2013): "Value and Momentum Everywhere"
- Lopez de Prado (2018): "The 10 Reasons Most Machine Learning Funds Fail"
- Bailey & Lopez de Prado (2014): "The Deflated Sharpe Ratio"
- Gu, Kelly, Xiu (2020): "Empirical Asset Pricing via Machine Learning"

### Open Source Tools

| Tool | Purpose | Notes |
|------|---------|-------|
| NautilusTrader | Full backtesting + live trading | Rust core, Python edge. Production-grade |
| VectorBT | Vectorized backtesting | Much faster than event-driven for ML |
| Backtrader | Event-driven backtesting | Feature-rich, already in requirements |
| mlfinlab | Lopez de Prado methods | Triple barrier, meta-labeling, CPCV, frac diff |
| Riskfolio-Lib | Portfolio optimization | Risk parity, HRP, multiple risk measures |
| pysystemtrade | Full system (Rob Carver) | Futures-focused, includes position sizing |
| fracdiff | Fractional differentiation | Single-purpose, easy to integrate |

---

## 10. Revised Recommendations for Ben10

Based on everything above, here is what changes or gets added to the project direction:

### Immediate High-Impact Additions

1. **Triple barrier labeling** — replace binary up/down with path-dependent labels (profit target, stop loss, time expiry). This single change will likely improve model quality more than adding any new architecture. Use `mlfinlab` or implement from scratch.

2. **Fractional differentiation** — apply to price series before feature computation. Maintains memory while achieving stationarity. Use `fracdiff` library.

3. **Meta-labeling** — after fixing existing models, implement a two-stage approach: primary model generates signals, secondary model filters. This is the most validated ensemble technique in financial ML.

4. **LightGBM and CatBoost** — add as direct comparisons to XGBoost. LightGBM is often faster with similar accuracy. CatBoost handles categorical features (day-of-week, trading session) natively.

5. **SHAP analysis** — before adding features, understand which of your 161+ indicators actually matter. Cut to 20–30 with economic reasoning.

### Architecture Evolution

6. **Regime detection** — implement as a preprocessing layer. Classify market state (trending up, trending down, range-bound, volatile) and either route to specialized models or include regime as a feature. Autoencoder + KMeans (already a stub) or Hidden Markov Models.

7. **TFT (Temporal Fusion Transformer)** — the single most valuable new architecture to add. Interpretable, handles mixed input types, strong empirical results.

8. **Foundation model integration** — use Chronos-2 or TimesFM as a feature generator. Feed their forecasts as additional features into your gradient boosting ensemble.

### Validation Overhaul

9. **CPCV** — replace simple train/test split with combinatorial purged cross-validation. Use `mlfinlab` implementation.

10. **Deflated Sharpe Ratio** — always report alongside standard Sharpe. Track how many strategies/parameters were tested.

11. **Monte Carlo robustness testing** — run 10,000 simulations on every strategy before trusting results.

### Gold-Specific Features

12. **Add macro features**: DXY, real yields (10Y minus CPI), VIX, gold ETF flows, COT positioning, EPU index. Gold is fundamentally a macro asset — pure technicals miss the biggest moves.

13. **Session-aware features**: Asian/London/NY session as categorical. Gold behaves differently across sessions.

14. **Event features**: FOMC, NFP, CPI dates as binary features. These cause the sharpest gold moves.

### Risk Management Layer

15. **Volatility targeting** — scale positions by inverse ATR so each trade has equal risk.

16. **Half-Kelly sizing** — use the Kelly criterion at 50% to balance growth and drawdown.

17. **Drawdown circuit breaker** — halt trading if drawdown exceeds threshold (e.g., 15%).

### What to Deprioritize

- **RL agents** — evidence doesn't support RL as primary approach for this use case. Implement eventually for completeness but don't expect it to be your edge.
- **Adding more DL architectures before fixing existing ones** — fix the 4 broken sequence models first.
- **Complex infrastructure** — you don't need co-location or sub-second latency for 4H trading.

---

*This research should be revisited quarterly as the field moves fast. The foundation model landscape in particular is evolving rapidly.*
