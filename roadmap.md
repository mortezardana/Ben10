# Ben10 — Development Roadmap

**Last updated:** 2026-03-19

This roadmap is a living document. Phases are sequential in priority but not strictly blocking — work on later phases can begin once the critical dependencies from earlier phases are met.

See `AGENT_EXECUTION_PLAN.md` for the detailed, parallelizable implementation plan with task-level specifications.
See `research_algorithmic_trading.md` for the full research backing these decisions.
See `first_claude_assessment.md` for the initial codebase assessment.

---

## Phase 0: Foundation Fixes

> Fix the issues that invalidate current results. Nothing else matters until these are addressed.

- [ ] **CRITICAL: Remove known leaking features** — `future_returns` and `signal` columns in `gold_4h.csv` contain future information. These MUST be excluded from all training immediately. This is the #1 reason XGBoost shows 100% accuracy followed by -100% backtest returns.
- [ ] **Fix normalization leakage** — fit scaler on training data only, transform val/test with training scaler, persist scaler with model
- [ ] **Fix sequence model inputs** — implement proper `(samples, timesteps, features)` windowing for LSTM, GRU, CRNN, CNN-LSTM using a shared utility
- [ ] **Audit indicator computation** — verify all remaining indicators in `gold_4h.csv` are strictly causal (no look-ahead). Recompute any that are suspect.
- [ ] **First-pass feature reduction** — immediately drop the 61 CDL (candlestick pattern) columns (very weak signal on 4H data), drop highly correlated pairs (>0.95). Target: reduce from 161 to ~40-50 features before any model training. Sophisticated selection (SHAP, RFE) comes in Phase 5.
- [ ] **Seed management** — create `set_all_seeds()` utility covering Python, NumPy, TensorFlow, PyTorch
- [ ] **Clean up empty stubs** — either implement or remove: `autoencoder.py`, `anomaly_detection.py`, `multitask_model.py`, `rl_agent.py`, `ensemble.py`, `metrics.py`, `helpers.py`, `trainer.py`
- [ ] **Implement triple barrier labeling** — replace binary up/down with path-dependent labels (take-profit, stop-loss, time expiry) using `mlfinlab` or custom implementation. Use volatility-scaled barriers (ATR-based). (See research: Lopez de Prado, "Advances in Financial ML")
- [ ] **Implement fractional differentiation** — apply to price-based features to achieve stationarity while retaining memory (d ≈ 0.2 retains >90% correlation with original series). Use `fracdiff` library.
- [ ] **CUSUM event-based sampling** — instead of labeling every 4H bar (most are noise), use a CUSUM filter to identify structurally meaningful events and only generate labels/predictions at those points. Reduces noise in labels significantly.

**Exit criteria:** All models produce valid, non-leaked results. Sequence models actually learn temporal patterns. Labels reflect realistic trade outcomes. Feature set is clean and reduced.

---

## Phase 1: Evaluation & Metrics Overhaul

> You can't improve what you can't measure properly. Accuracy alone is meaningless — a 55% accurate model can lose money if it's right on small moves and wrong on big ones.

- [ ] **Implement `utils/metrics.py`** with trading-specific metrics:
  - Profit factor (gross profit / gross loss)
  - Max drawdown (peak-to-trough equity decline)
  - Calmar ratio (annualized return / max drawdown)
  - Sortino ratio (return / downside deviation — better than Sharpe for asymmetric returns)
  - Win rate + avg win/loss ratio
  - Expectancy: (win_rate × avg_win) − (loss_rate × avg_loss)
  - Trade count, max consecutive losses
  - Statistical significance (binomial test, bootstrap confidence intervals)
  - Deflated Sharpe Ratio — corrects for multiple testing and non-normal returns (Bailey & Lopez de Prado, 2014)
- [ ] **Add benchmark strategies:**
  - Buy and hold gold
  - Random entry/exit (Monte Carlo baseline — 10,000 simulations)
  - Simple moving average crossover (e.g., 50/200 SMA)
- [ ] **Standardize evaluation output** — every model run produces a consistent metrics JSON/dict that can be compared
- [ ] **SHAP analysis** — implement SHAP value extraction for XGBoost to understand which of the 161+ features carry actual predictive power. Use this to guide feature selection in Phase 5.

**Exit criteria:** Every model is evaluated on 10+ trading metrics, compared against benchmarks, and statistical significance is reported.

---

## Phase 2: Backtesting Engine

> Replace the 24-line backtest with something that reflects reality.

- [ ] **Integrate vectorbt or backtrader** as the backtesting backend
- [ ] **Add transaction costs** — configurable spread and commission
- [ ] **Add slippage model** — at minimum, next-bar-open execution
- [ ] **Support long and short positions** — act on both 1 and 0 predictions
- [ ] **Position sizing** — fixed fractional, Kelly criterion, or volatility-scaled
- [ ] **Stop-loss and take-profit** — configurable per-trade risk limits
- [ ] **Trade-level logging** — entry/exit price, duration, P&L per trade
- [ ] **Drawdown tracking** — real-time max drawdown and drawdown duration

**Exit criteria:** Backtest produces realistic P&L with costs, slippage, and risk management. Individual trades can be inspected.

---

## Phase 3: Validation Framework

> Prove results generalize across market conditions. A single train/test split is one snapshot — gold has gone through bull runs (2004–2011), range-bound (2013–2018), and volatile breakouts (2020–2026). Your model must survive all of them.

- [ ] **Walk-forward validation** — rolling train/test windows across the full dataset. Report Walk-Forward Efficiency (OOS return / IS return) — values >70% indicate transferability.
- [ ] **Expanding window validation** — growing training set with fixed test windows
- [ ] **Combinatorial Purged Cross-Validation (CPCV)** — Lopez de Prado's method: multiple chronology-respecting partitions with purging (remove near-boundary observations) and embargo (gap after purging). Use `mlfinlab` implementation. Shows "marked superiority in mitigating overfitting risks."
- [ ] **Regime-aware splits** — ensure train/test windows span different market conditions
- [ ] **Out-of-sample reporting** — report performance distribution (mean, std, worst-case) across all windows
- [ ] **Overfitting detection** — compare in-sample vs out-of-sample performance gap. Compute Probability of Backtest Overfitting (PBO).
- [ ] **Monte Carlo robustness testing** — 10,000+ simulations with reshuffled trade sequences, randomized parameters, skipped trades, and resampled returns

**Exit criteria:** Model performance is reported as a distribution across multiple market periods. PBO < 0.5. Monte Carlo simulations show consistent performance.

---

## Phase 4: Pipeline Architecture & Engineering

> Restructure the system into independent pipelines with a shared infrastructure layer and a parent combiner. This is the "alpha factory" pattern used by quant firms — not microservices, but a modular monolith with clear boundaries.
>
> See `research_architecture.md` for the full research on why modular monolith > microservices for trading systems.

### 4.1 Core Interface & Pipeline Abstraction
- [ ] **Define `TradingPipeline` interface** — the single most impactful architectural decision. Every model family implements: `train(data) -> None`, `predict(data) -> PipelineOutput`, `evaluate(data) -> dict`. `PipelineOutput` is a standardized dataclass: `signals` (Series of -1/0/+1), `confidence` (Series of 0.0-1.0), `metadata` (dict with model name, version, params).
- [ ] **All pipelines must output calibrated probabilities** — not just binary 0/1 predictions. The confidence score (0.0-1.0) must reflect true probability, not just model softmax output. Use Platt scaling (logistic calibration) or isotonic regression on a held-out calibration set. Without this, the parent algorithm cannot meaningfully compare or weight signals across pipelines. A "0.7 from XGBoost" must mean the same thing as "0.7 from LSTM."
- [ ] **Wrap existing models as pipelines:**
  - `pipelines/gradient_boosting/` — XGBoost, LightGBM, CatBoost. Flat feature vector, SHAP interpretability. Expected to be the strongest pipeline.
  - `pipelines/deep_learning/` — LSTM, GRU, CNN-LSTM, TCN, Transformer, TFT. Sequence input, temporal patterns.
  - `pipelines/regime/` — HMM, autoencoder, anomaly detection. Outputs regime labels + confidence, not direct trading signals. Feeds into other pipelines and the combiner.
  - `pipelines/experimental/` — RL, foundation models, alternative approaches. Lower priority, sandbox for new ideas.
- [ ] **Shared infrastructure layer (`shared/`):**
  - `shared/data_loader.py` — single source of truth for all data loading
  - `shared/feature_store.py` — compute features once, cache as Parquet, serve to all pipelines (read-only). Graduate to Feast when online serving is needed.
  - `shared/metrics.py` — standardized evaluation (from Phase 1)
  - `shared/backtest.py` — common backtesting engine (from Phase 2)
- [ ] **Pipeline independence rules:**
  - Each pipeline has its own `config.yaml` and hyperparameters
  - No pipeline can access another pipeline's internal state
  - Shared infrastructure is read-only for pipelines
  - Communication between pipelines only through published `PipelineOutput` objects

### 4.2 The Combiner (Parent Algorithm)
> Each pipeline produces a weighted confidence signal ("I'm 72% confident this is bullish"). The parent algorithm gathers all these weighted signals and synthesizes them into a single, more confident trading decision with a confidence-based position size. This is the proven approach — WorldQuant built $9B+ AUM on combining millions of weak signals into one strong signal.

- [ ] **Level 1 — Weighted voting** — weight each pipeline by rolling out-of-sample accuracy (not equal votes). Auto-downweights underperformers.
- [ ] **Level 2 — Stacking meta-learner** — XGBoost trained on pipeline probability outputs + market features. Learns *when* each pipeline is reliable.
- [ ] **Level 3 — Meta-labeling** — best pipeline picks direction, meta-model decides whether to act. Probability output = position size.
- [ ] **Level 4 — Regime-conditional routing** — HMM regime state determines which pipeline(s) get highest weight. (e.g., "trending market → trust gradient boosting momentum signals; choppy market → trust regime pipeline, reduce all sizes")
- [ ] **Level 5 — Mixture of Experts (future)** — gating network learns to route dynamically based on market features.
- [ ] **CRITICAL: Parent trains only on out-of-sample child predictions** — if the parent sees in-sample predictions from child models, it learns their overfit patterns, not their real skill. Implementation: use walk-forward or nested cross-validation — child models predict on folds they weren't trained on, parent trains on those honest predictions only.
- [ ] **Signal independence tracking** — continuously measure pairwise correlation between pipeline signals. Feed this into the parent's weighting logic: uncorrelated signals get higher combined weight (they provide independent information), correlated signals get downweighted (they're redundant). Three uncorrelated 53% signals combined > three correlated 56% signals.
- [ ] **Disagreement handling** — when pipelines disagree: reduce position size proportionally OR sit out entirely. Research shows agreement filtering produces more stable signals.

### 4.3 Configuration
- [ ] **Hierarchical config system** — YAML or Pydantic-based. Global config + per-pipeline config that inherits/overrides.
- [ ] **CLI argument parsing** — `main.py --pipeline gradient_boosting --config experiments/xgb_v2.yaml` or `main.py --run-all` to orchestrate all pipelines + combiner.
- [ ] **Environment-aware config** — dev/test/production settings

### 4.4 Experiment Tracking
- [ ] **MLflow integration** — lightweight (`pip install mlflow`), immediately useful. Each pipeline logs to its own MLflow experiment. The combiner logs to a separate experiment comparing pipeline outputs.
- [ ] **Model registry** — MLflow Model Registry with staging/production states per pipeline. Combiner checks registry for current production model from each pipeline.
- [ ] **Comparison tooling** — cross-pipeline performance comparison dashboard

### 4.5 Orchestration & Parallelism
- [ ] **Sequential orchestrator (start here)** — `main.py` runs each pipeline sequentially, collects `PipelineOutput` objects, feeds to combiner. No infrastructure needed.
- [ ] **Ray for parallel training (when needed)** — when training all pipelines sequentially becomes too slow. Ray's shared-memory object store loads data once, all pipelines access it. ~10 lines of code to parallelize.
- [ ] **Do NOT add** Kafka, Kubernetes, gRPC, Celery, or multi-repo until Phase 10 at earliest. These add complexity with zero benefit at current scale.

### 4.6 Reproducibility
- [ ] **Full seed control** across all libraries
- [ ] **Environment pinning** — exact dependency versions in lockfile
- [ ] **Data versioning** — hash and track dataset versions

**Exit criteria:** Any experiment can be reproduced exactly. Pipelines are independent and implement a standard interface. Combiner produces a single trading decision from all pipeline outputs. MLflow tracks all experiments.

---

## Phase 5: Feature Engineering

> Better inputs lead to better predictions.

### 5.1 Feature Selection
- [ ] **Correlation filtering** — remove features with >0.95 pairwise correlation
- [ ] **Importance-based selection** — use XGBoost/SHAP to rank, keep top-N (target: 20–30 features with economic reasoning, cap at 30-40 total across all timeframes)
- [ ] **Recursive feature elimination** — iteratively remove weakest features
- [ ] **Per-model feature sets** — different models may benefit from different features
- [ ] **PCA / Autoencoder dimensionality reduction** — as alternative to manual selection

### 5.2 Feature Computation
- [ ] **In-pipeline indicator calculation** — recompute indicators from raw OHLCV inside the pipeline (eliminates leakage risk)
- [ ] **Feature engineering module** — centralized, testable, versionable indicator computation
- [ ] **Volume-based indicators** — compute from existing OHLCV data (no new data needed):
  - OBV (On-Balance Volume) + slope and price divergence detection
  - MFI (Money Flow Index, 14-period) — volume-weighted RSI
  - A/D (Accumulation/Distribution) line + slope
  - VWAP (reset daily, primarily for 1H features)
  - Volume profile: POC (Point of Control), Value Area, distance-from-POC
- [ ] **See `research_order_flow.md`** for detailed formulas and feature definitions

### 5.3 Target Engineering
- [ ] **Triple barrier labels** (if not done in Phase 0) — path-dependent labeling with take-profit, stop-loss, and time expiry barriers
- [ ] **Meta-labeling** — secondary model that decides whether to act on primary model's signal (trade or pass). Dramatically reduces false positives.
- [ ] **3-class target** — up / flat / down with configurable threshold
- [ ] **Regression target** — predict return magnitude, threshold for trading
- [ ] **Volatility-adjusted target** — relative to ATR
- [ ] **Multi-horizon targets** — 1-bar, 4-bar, 12-bar predictions

### 5.4 Multi-Timeframe Data & Features
> Research shows 3 timeframes is optimal. The 4:1 ratio rule (Elder) provides enough separation for different structure without redundancy. Going below 1H adds noise, not signal. Going beyond 4 timeframes has diminishing returns.

#### 5.4.1 Data Acquisition
- [ ] **Acquire 1H gold data** (2004–2025) — ~6,200 bars/year, ~5.4 MB for 20 years. Entry timing and session dynamics.
- [ ] **Acquire Daily gold data** (2004–2025) — ~252 bars/year. Trend filter and institutional anchor levels. **Highest value addition.**
- [ ] **Acquire Weekly gold data** (2004–2025) — ~52 bars/year. Macro context only, 2-3 features.
- [ ] **Do NOT acquire sub-hourly data** (15min, 5min, 1min) — research confirms these add noise, not signal, for a 4H prediction horizon. Overfitting risk increases, out-of-sample performance degrades.

#### 5.4.2 Timeframe Alignment (CRITICAL — prevents look-ahead bias)
- [ ] **Replace outer join with `pd.merge_asof(direction='backward')`** — the current `merge_multi_timeframes()` uses an outer join that can leak future higher-TF data. A 4H bar at 08:00 UTC must NOT see today's daily close (not yet available). `merge_asof` with backward direction ensures each row only sees completed higher-TF bars.
- [ ] **Shift higher-TF features by one period before merging** — `daily_features.shift(1)` so a 4H bar always uses yesterday's completed daily bar. Same for weekly.
- [ ] **Standardize timezone handling** — all data in UTC. Gold 4H bar boundaries vary by broker. Pick one convention and stick to it.
- [ ] **Handle weekend/holiday gaps explicitly** — drop weekend bars, don't forward-fill across gaps.

#### 5.4.3 Multi-Timeframe Feature Strategy
> Don't dump all columns from all timeframes. Be selective. Cap total features at 30-40 across all timeframes.

- [ ] **Daily features (5-8 features, highest impact):**
  - `daily_trend`: is Daily SMA_50 > SMA_200? (binary)
  - `daily_atr`: current daily volatility
  - `daily_rsi`: oversold/overbought on daily scale
  - `daily_ma_distance`: distance of price from daily 50 MA (normalized)
  - `daily_close_vs_open`: bullish or bearish daily candle (binary)
  - `daily_range_position`: where current price sits within daily high-low range (0-1)
- [ ] **Weekly features (2-3 features):**
  - `weekly_trend`: is Weekly SMA_20 > SMA_50? (binary)
  - `weekly_ma_position`: price above/below weekly 50 MA (binary)
  - `weekly_atr`: weekly volatility for context
- [ ] **1H-derived features (3-5 features, aggregated per 4H bar):**
  - `session_indicator`: Asian/London/NY overlap as categorical (gold's strongest intraday pattern)
  - `intra_bar_trend`: direction of 1H bars within the 4H bar (trending or choppy)
  - `intra_bar_volatility`: volatility of 1H bars within each 4H bar
  - `london_open_momentum`: price change in the first 1H of London session
  - `volume_profile`: how volume is distributed within the 4H bar (front-loaded vs back-loaded)

#### 5.4.4 Multi-Timeframe Model Architecture
> Three approaches in order of sophistication. Start with Approach A.

- [ ] **Approach A: Selective higher-TF features (do first)** — add the Daily/Weekly/1H features above directly to the 4H feature set. Simple, works with all existing models. Expected improvement: 3-8%.
- [ ] **Approach B: Per-timeframe models + stacking meta-learner** — train XGBoost per timeframe, feed probability outputs (not binary 0/1) + cross-TF features into a final meta-model. Upgrade from current simple voting. Expected improvement over A: 2-5%.
- [ ] **Approach C: Hierarchical architecture** — Daily model outputs feed as input features to 4H model. 4H model learns to weight daily context. Top-down knowledge flow only (higher → lower), preventing noise propagation upward.
- [ ] **Fix existing multi-TF code:**
  - Upgrade `utils/multi_timeframe_utils.py` to use `merge_asof` + shift
  - Upgrade `utils/signal_voting.py` to support 3+ timeframes and probability-weighted voting
  - Integrate multi-TF into `main.py` (currently unused)
  - Fix dashboard to work with actual available data files

### 5.5 External Data (HIGH IMPACT — consider pulling earlier)
> Gold is fundamentally a macro asset. Pure technicals miss the biggest moves. Real yields, DXY, and central bank buying are the primary drivers. Even adding just DXY + 10Y yield can meaningfully improve predictions.

- [ ] **USD Index (DXY)** — strong inverse correlation with gold
- [ ] **US Treasury yields (2Y, 10Y, real yields)** — real yields are the single strongest gold driver
- [ ] **VIX / S&P 500** — risk appetite indicators
- [ ] **Oil prices (WTI/Brent)** — commodity correlation
- [ ] **Macro calendar events** (FOMC, NFP, CPI) — these cause the sharpest gold moves
- [ ] **COT report data** (institutional positioning in gold futures) — free from CFTC weekly. Key features: net speculative positioning, week-over-week change, extreme positioning flag (>90th percentile predicts reversals). See `research_order_flow.md`.
- [ ] **COMEX open interest** — free daily from CME. Rising OI + rising price = strong trend; falling OI + rising price = weak rally (short covering). Divergence detection is high value.
- [ ] **Gold options data** (put/call ratio, implied volatility, IV percentile, skew) — from CME/Barchart. IV is forward-looking volatility estimate, useful for both signals and position sizing. Extreme put/call ratios have contrarian value.
- [ ] **Retail sentiment (OANDA order book)** — free API, contrarian indicator. When >75% retail is long, gold statistically more likely to fall. Low weight but easy to add.
- [ ] **Gold ETF flows (GLD, IAU)** — large flows move price
- [ ] **Economic Policy Uncertainty Index (EPU)** — macro uncertainty drives gold demand
- [ ] **News/FOMC sentiment** — NLP on FOMC statements, gold-related news

**Exit criteria:** Pipeline computes its own features from raw data. Feature selection is automated. Multi-timeframe alignment is leak-free. External data enriches predictions. Total features capped at 30-40.

---

## Phase 6: Advanced Models

> Fill in the remaining model slots and add new architectures.

### 6.1 High Priority — Complete Existing Stubs
- [ ] **HMM regime detection** — 3-state Hidden Markov Model on returns + volatility to classify market regime (bull/quiet, choppy/uncertain, bear/crisis). Use `hmmlearn`. Feed regime as feature to all models AND use for position sizing (reduce size in choppy regimes). This is the proven approach — more reliable than autoencoders for regime detection.
- [ ] **Autoencoder** — for anomaly features and alternative regime detection. Compare with HMM approach.
- [ ] **Anomaly detection** — Isolation Forest / One-Class SVM for unusual market conditions. Use to flag periods where model predictions should not be trusted.
- [ ] **Multitask model** — shared encoder with three heads: direction (classification), magnitude (regression for position sizing), volatility (regression for risk management). Joint training improves both tasks by 2-5%.

### 6.2 High Priority — New Architectures
- [ ] **LightGBM** — direct comparison with XGBoost. Often faster with similar accuracy. Add immediately.
- [ ] **CatBoost** — handles categorical features (day-of-week, session, regime) natively
- [ ] **Temporal Fusion Transformer (TFT)** — most valuable new architecture. Interpretable attention, variable selection networks, handles mixed input types. Prioritize this.

### 6.3 Medium Priority — Additional Architectures
- [ ] **Informer / Autoformer** — efficient transformer variants for long sequences (ProbSparse attention)
- [ ] **PatchTST** — patches time series before attention, up to 21% lower MSE
- [ ] **N-BEATS / N-HiTS** — neural basis expansion for time series
- [ ] **WaveNet-style models** — dilated causal convolutions

### 6.4 Lower Priority
- [ ] **RL agent** — PPO/DQN via Stable-Baselines3. Evidence does not strongly support RL for position-taking strategies at this frequency. Implement for completeness but don't expect it to be the primary edge.
- [ ] **Foundation model integration** — use Chronos-2 or TimesFM as feature generators. Feed their forecasts as additional features into gradient boosting ensemble. Don't use as standalone trading systems.

### 6.5 Model Improvements
- [ ] **Bidirectional LSTM/GRU variants**
- [ ] **Attention layers on top of RNNs**
- [ ] **Transformer with positional encoding tuning**
- [ ] **TabNet attention mask extraction** for interpretability

**Exit criteria:** 12+ model types available. Each model has a clear use case and documented strengths.

---

## Phase 7: Ensemble Refinement & Signal Intelligence

> Phase 4's combiner provides the basic combination. This phase makes it sophisticated. The goal: trade less, trade better. WorldQuant's 4 million alphas have average pairwise correlation of only 15.9% — diversity is the source of edge.

### 7.1 Combiner Upgrades (builds on Phase 4.2)
- [ ] **Meta-labeling ensemble** — primary pipeline generates signals, meta-learner decides whether to act. Meta-model probability output directly becomes bet size (0 = no trade, 1 = full size). Most validated approach in financial ML (Lopez de Prado).
- [ ] **Confidence thresholding** — only trade when combined confidence exceeds threshold (e.g., >60%). Research shows this doubles Sharpe ratio while cutting trade frequency by 80-90%.
- [ ] **Dynamic model selection** — track per-pipeline rolling performance, upweight recent winners, downweight underperformers. Use exponentially weighted moving average of out-of-sample accuracy.

### 7.2 Pipeline Diversity
- [ ] **Ensemble diversity metrics** — measure prediction correlation between all pipeline pairs. If two pipelines have >0.8 correlation, they're redundant — keep the better one or force feature differentiation.
- [ ] **Forced diversity** — ensure pipelines use genuinely different approaches: different features, different model families, different timeframes, different target definitions. Correlated pipelines waste compute without adding signal.

### 7.3 Uncertainty & Knowing When Not to Trade
- [ ] **MC Dropout** — run DL inference N times with dropout, measure variance. High variance = model is unsure = reduce size or sit out.
- [ ] **Quantile regression** — predict 10th/50th/90th percentile. Only trade when entire interval is on one side of zero.
- [ ] **Conformal prediction (CQR)** — distribution-free prediction intervals with coverage guarantees.
- [ ] **Disagreement as signal** — when pipelines strongly disagree, this itself is information. Log and analyze — are disagreement periods predictably bad?

**Exit criteria:** Ensemble consistently outperforms the best individual pipeline across walk-forward windows. System knows when it doesn't know and sits out. Pipeline diversity is measured and maintained.

---

## Phase 8: Risk Management

> The difference between a model and a trading system. Position sizing and risk control matter as much as signal quality.

- [ ] **Volatility targeting** — scale positions inversely to ATR so each trade has equal risk contribution. Target specific portfolio volatility (e.g., 10% annualized).
- [ ] **Half-Kelly position sizing** — use Kelly criterion at 50% to balance growth and drawdown. Full Kelly produces extreme drawdowns; quarter Kelly is most conservative.
- [ ] **Fixed fractional sizing** — alternative to Kelly: risk X% of equity per trade
- [ ] **Per-trade risk limits** — max loss per trade as % of equity
- [ ] **Portfolio-level limits** — max drawdown threshold (e.g., 15%), daily loss limit, exposure caps
- [ ] **Correlation-aware sizing** — reduce size when adding correlated positions (relevant for multi-asset Phase 11)
- [ ] **Drawdown circuit breaker** — halt trading if drawdown exceeds threshold, require manual review before resuming
- [ ] **VaR / CVaR computation** — Value at Risk and Conditional VaR for risk reporting

**Exit criteria:** No trade risks more than X% of equity. System halts automatically under extreme drawdown. Risk metrics computed and logged for every trading session.

---

## Phase 9: Live Data & Paper Trading

> Bridge the gap between research and reality.

- [ ] **Data provider integration** — real-time or near-real-time candle feeds for all 4 timeframes (1H, 4H, Daily, Weekly) from broker API or data vendor. OANDA, MT5, or Interactive Brokers all support multi-timeframe historical + streaming data for gold.
- [ ] **Feature computation on live data** — same indicator pipeline and multi-timeframe alignment used in training. Must use the same `merge_asof` + shift logic to prevent training-serving skew.
- [ ] **Inference pipeline** — load model, compute features, generate prediction, apply ensemble
- [ ] **Paper trading mode** — simulate trades in real-time without real capital
- [ ] **Performance tracking** — live equity curve, metrics dashboard, comparison to backtest expectations
- [ ] **Alerting** — notify on high-confidence signals (email, Telegram, webhook)

**Exit criteria:** System runs autonomously in paper trading mode, generating signals and tracking simulated P&L in real-time.

---

## Phase 10: Production & Deployment

> Only after paper trading validates the system.

- [ ] **Broker integration** — order execution via API (e.g., OANDA, Interactive Brokers, MetaTrader). Consider NautilusTrader for research-to-live parity.
- [ ] **Order management** — handle fills, partial fills, rejections, requotes
- [ ] **Model monitoring** — detect performance degradation / concept drift per pipeline. Alert when any pipeline's rolling accuracy drops below threshold.
- [ ] **Automated retraining** — scheduled retraining with latest data. Each pipeline retrains independently. Combiner re-evaluates pipeline weights.
- [ ] **Containerization** — Docker Compose: app + MLflow + TimescaleDB. Add Redis for real-time signal pub/sub only if needed.
- [ ] **Scheduling** — cron / Airflow / Prefect for periodic inference and retraining
- [ ] **Failover & recovery** — handle disconnections, crashes, partial state
- [ ] **Audit trail** — log every pipeline output, combiner decision, and trade for review. Full traceability from signal to execution.

**Exit criteria:** System runs in production with real capital, monitored, with automated failsafes.

---

## Phase 11: Multi-Asset Expansion

> Validate that the approach generalizes.

- [ ] **Silver (XAG/USD)** — closest analog to gold
- [ ] **Forex majors** (EUR/USD, GBP/USD) — test on different market dynamics
- [ ] **Oil (WTI/Brent)** — another commodity
- [ ] **Cross-asset signals** — use one asset's prediction to inform another
- [ ] **Portfolio-level management** — trade multiple assets with correlation-aware sizing

**Exit criteria:** Pipeline works on multiple assets with minimal configuration changes. Portfolio approach outperforms single-asset.

---

## Ongoing / Cross-Cutting

These are not phases — they apply continuously:

- [ ] **Documentation** — keep README, assessment, and roadmap current
- [ ] **Testing** — unit tests for data pipeline, metrics, feature computation
- [ ] **Code quality** — type hints, consistent interfaces, linting
- [ ] **Performance profiling** — training speed, memory usage, inference latency
- [ ] **Research log** — document experiments, findings, dead ends (what didn't work is as valuable as what did)
- [ ] **Alpha decay monitoring** — track strategy performance over time. Medium-frequency momentum signals typically decay 60% within 10 months. Plan for periodic retraining.
- [ ] **Multiple testing discipline** — always report how many strategies/parameters were tested alongside results. Use Deflated Sharpe Ratio.

---

## Key Resources

### Essential Reading
1. **"Advances in Financial Machine Learning"** — Lopez de Prado (triple barrier, meta-labeling, frac diff, CPCV)
2. **"Machine Learning for Algorithmic Trading" (2nd ed.)** — Stefan Jansen (comprehensive, code-focused)
3. **"Quantitative Trading"** — Ernest Chan (2025 edition)

### Key Libraries
| Library | Purpose |
|---------|---------|
| `mlfinlab` | Triple barrier, meta-labeling, CPCV, frac diff |
| `fracdiff` | Fractional differentiation |
| `shap` | Feature importance / interpretability |
| `vectorbt` | Vectorized backtesting |
| `riskfolio-lib` | Portfolio optimization, risk parity |
| `hmmlearn` | HMM regime detection |
| `NautilusTrader` | Production-grade backtesting + live trading |

### Key Research
See `research_algorithmic_trading.md` for the full algorithmic trading research report.
See `research_multi_timeframe.md` for multi-timeframe analysis research and architecture guidance.
See `research_architecture.md` for the modular pipeline architecture research (alpha factory, combiner patterns, infrastructure decisions).
See `research_order_flow.md` for order flow and liquidity analysis research (why raw order book doesn't work for 4H gold, viable volume-based alternatives).

---

*This roadmap is intentionally ambitious. Not everything needs to be done. The goal is to have a clear path forward so effort is always directed at the highest-impact next step.*
