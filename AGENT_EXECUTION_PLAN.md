# Ben10 — Agent Execution Plan

**Created:** 2026-03-20
**Purpose:** Detailed, parallelizable implementation plan for a team of agents to execute simultaneously.

**Reference Documents:**
- `roadmap.md` — strategic roadmap
- `first_claude_assessment.md` — initial codebase assessment
- `research_algorithmic_trading.md` — ML trading research
- `research_architecture.md` — system architecture decisions
- `research_multi_timeframe.md` — multi-timeframe analysis
- `research_order_flow.md` — order flow / volume analysis

---

## How This Plan Works

Tasks are organized into **Waves**. All tasks within a wave can run **in parallel** — they have no dependencies on each other. A wave must complete before the next wave begins (unless noted otherwise).

Each task specifies:
- **What:** Exactly what to build
- **Where:** Files to create or modify
- **Inputs:** What the agent needs to read/understand
- **Outputs:** What the agent must produce (acceptance criteria)
- **Tests:** How to verify correctness
- **References:** Which research docs contain relevant detail

---

## Current Codebase State (READ THIS FIRST)

```
Ben10/
├── main.py                          # Orchestrator — trains all 8 models sequentially
├── pipeline/
│   ├── config.py                    # CONFIG dict (paths, XGB params, split ratios)
│   ├── data_loader.py               # load_data(), normalize_features() — HAS LEAKAGE
│   ├── backtest.py                  # simple_strategy_backtest() — 25 lines, inadequate
│   ├── evaluator.py                 # Plotting functions only
│   └── trainer.py                   # STUB (1 line)
├── models/
│   ├── baseline_ml.py               # XGBoost — WORKS but trains on leaked data
│   ├── lstm_model.py                # BROKEN: reshapes as (samples, features, 1)
│   ├── cnn_lstm.py                  # BROKEN: same reshape issue
│   ├── crnn_model.py                # BROKEN: same reshape issue
│   ├── gru_model.py                 # BROKEN: same reshape issue
│   ├── tcn_model.py                 # WORKS: proper (samples, timesteps, features) windowing
│   ├── transformer_model.py         # WORKS: proper windowing
│   ├── tabnet_model.py              # WORKS: flat input (correct for TabNet)
│   ├── multi_timeframe_models.py    # Per-TF XGBoost training
│   ├── predictor.py                 # Model loading utility
│   ├── ensemble.py                  # STUB
│   ├── anomaly_detection.py         # STUB
│   ├── autoencoder.py               # STUB
│   ├── multitask_model.py           # STUB
│   └── rl_agent.py                  # STUB
├── utils/
│   ├── logger.py                    # Logging with file+console handlers
│   ├── plot_utils.py                # Timestamped plot directories
│   ├── signal_voting.py             # 2-timeframe signal voting (confirm/filter/majority)
│   ├── multi_timeframe_utils.py     # merge_multi_timeframes() — HAS LOOK-AHEAD BIAS (outer join)
│   ├── metrics.py                   # STUB
│   └── helpers.py                   # STUB
├── dashboard/                       # Streamlit UI (main.py, controller.py, pages/)
├── Data/
│   └── gold_4h.csv                  # 161 columns, ~16K rows, 2004-2025. CONTAINS: future_returns, signal (LEAK)
└── requirements.txt                 # deps already include: xgboost, lightgbm, tensorflow, torch, vectorbt, backtrader
```

**Critical known issues:**
1. `gold_4h.csv` contains `future_returns` and `signal` columns — these leak the target variable
2. `normalize_features()` fits scaler on full dataset before split — data leakage
3. LSTM, GRU, CRNN, CNN-LSTM reshape as `(samples, features, 1)` instead of `(samples, timesteps, features)` — they can't learn temporal patterns
4. Backtest is 25 lines — no costs, no slippage, no shorts, no position sizing
5. Only `gold_4h.csv` exists — no other timeframe data
6. `multi_timeframe_utils.py` uses outer join — look-ahead bias

---

# WAVE 1 — Foundation Fixes (All Parallel)

> Fix everything that makes current results invalid. These 7 tasks have ZERO dependencies on each other.

---

### Task 1.1: Remove Data Leakage from Dataset & Loader

**What:** Remove future-leaking columns from the data loading pipeline and audit the CSV for other leakage sources.

**Where:**
- MODIFY: `pipeline/data_loader.py`
- MODIFY: `pipeline/config.py` (add leaking columns to exclude list)
- DO NOT modify `Data/gold_4h.csv` directly — filter in code so the raw data is preserved

**Steps:**
1. Read `pipeline/data_loader.py` and `pipeline/config.py`
2. Add `future_returns` and `signal` to `CONFIG['exclude_cols']`
3. In `load_data()`, ensure these columns are dropped BEFORE any processing
4. Read `Data/gold_4h.csv` headers — audit ALL 161 column names for any other column that sounds like it contains future information (e.g., anything with "future", "forward", "next", "target" in the name)
5. Document all excluded columns and why in a code comment
6. Add a validation check: if any column name contains suspicious keywords, log a warning

**Outputs:**
- `data_loader.py` that explicitly excludes all leaking columns before any processing
- Config updated with full exclusion list
- Code comment listing every excluded column and the reason

**Tests:**
- After loading data, assert `future_returns` and `signal` are NOT in the DataFrame columns
- Assert no column name contains "future", "forward", "next" (case-insensitive)

---

### Task 1.2: Fix Normalization Leakage

**What:** Fix `normalize_features()` so the scaler is fitted ONLY on training data, then applied to val/test. Persist the scaler alongside the model.

**Where:**
- MODIFY: `pipeline/data_loader.py`

**Steps:**
1. Read current `normalize_features()` — it currently fits `StandardScaler` on the entire DataFrame
2. Change the function signature to accept split indices or a mode parameter
3. Implement: fit scaler on `df[:train_end]`, transform all splits with that same scaler
4. Add `save_scaler(scaler, path)` and `load_scaler(path)` functions using joblib
5. Return the fitted scaler alongside the normalized data so it can be persisted
6. Update all callers in `main.py` to use the new interface

**Outputs:**
- `normalize_features()` that takes train/val/test boundaries and only fits on training portion
- Scaler persistence functions
- Updated `main.py` calls

**Tests:**
- Fit scaler on training data, verify `scaler.mean_` matches training set statistics (not full dataset)
- Verify val/test are transformed using training scaler (not their own statistics)
- Round-trip test: save scaler, load scaler, verify identical transforms

---

### Task 1.3: Fix Sequence Model Inputs

**What:** Implement proper `(samples, timesteps, features)` windowing for LSTM, GRU, CRNN, CNN-LSTM. Create a shared utility so all sequence models use the same windowing logic.

**Where:**
- CREATE: `utils/sequence_utils.py`
- MODIFY: `models/lstm_model.py`
- MODIFY: `models/gru_model.py`
- MODIFY: `models/crnn_model.py`
- MODIFY: `models/cnn_lstm.py`

**Steps:**
1. Read `models/tcn_model.py` — it already has correct `create_sequences()` implementation. Use this as the reference.
2. Create `utils/sequence_utils.py` with a shared `create_sequences(data, target, sequence_length=32)` function that returns `(X, y)` where X has shape `(samples, sequence_length, features)`
3. Read each broken model file. Identify the reshape line (look for `.reshape(...)` with wrong dimensions)
4. Replace the broken reshape with a call to the shared `create_sequences()`
5. Update model input shapes to expect `(sequence_length, n_features)` instead of `(n_features, 1)`
6. Default `sequence_length=32` (32 bars × 4H = ~5 trading days of context)
7. Also update `predict_*` functions to use the same windowing

**Reference implementation (from `tcn_model.py`):**
```python
def create_sequences(data, target, sequence_length):
    X, y = [], []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i])
        y.append(target[i])
    return np.array(X), np.array(y)
```

**Outputs:**
- `utils/sequence_utils.py` with shared windowing function
- All 4 broken models use the shared utility
- All 4 models accept input shape `(batch, timesteps, features)`

**Tests:**
- Verify output shape: `X.shape == (n_samples, 32, n_features)` and `y.shape == (n_samples,)`
- Verify no data leakage: `X[i]` should contain data from rows `[i, i+sequence_length)` and `y[i]` should be the target at row `i+sequence_length`
- Verify each model can train for 1 epoch without shape errors
- Verify predictions have the correct length (total rows - sequence_length)

---

### Task 1.4: Seed Management Utility

**What:** Create a `set_all_seeds()` function that sets seeds for Python, NumPy, TensorFlow, and PyTorch. Use it at the start of every run.

**Where:**
- CREATE: `utils/seed.py`
- MODIFY: `main.py` (add seed call at start)

**Steps:**
1. Create `utils/seed.py` with `set_all_seeds(seed=42)` that sets:
   - `random.seed(seed)`
   - `np.random.seed(seed)`
   - `os.environ['PYTHONHASHSEED'] = str(seed)`
   - `tf.random.set_seed(seed)` (guard with try/except for import)
   - `torch.manual_seed(seed)` + `torch.cuda.manual_seed_all(seed)` (guard with try/except)
   - `torch.backends.cudnn.deterministic = True` (if torch available)
2. Add `set_all_seeds(CONFIG['random_state'])` as first call in `main.py`'s `main()`

**Outputs:**
- `utils/seed.py` that handles all random libraries
- `main.py` calls it at startup

**Tests:**
- Run `set_all_seeds(42)` twice, generate random numbers from numpy/python, verify identical

---

### Task 1.5: Trading Metrics Module

**What:** Implement a comprehensive trading metrics module. This is a standalone utility with NO dependencies on other tasks.

**Where:**
- REPLACE: `utils/metrics.py` (currently a stub)

**Steps:**
1. Implement all of the following functions that accept a DataFrame with columns `returns`, `positions`, `equity` (standard backtest output):

```
Core Metrics:
- profit_factor(trades) → float                    # gross_profit / gross_loss
- max_drawdown(equity) → float                     # peak-to-trough % decline
- max_drawdown_duration(equity) → int              # bars in longest drawdown
- calmar_ratio(equity) → float                     # annualized_return / max_drawdown
- sortino_ratio(returns, target=0) → float         # return / downside_deviation
- sharpe_ratio(returns, rf=0) → float              # (mean_return - rf) / std_return
- win_rate(trades) → float                         # % profitable trades
- avg_win_loss_ratio(trades) → float               # mean(wins) / mean(losses)
- expectancy(trades) → float                       # (win_rate * avg_win) - (loss_rate * avg_loss)
- trade_count(trades) → int
- max_consecutive_losses(trades) → int
- annualized_return(equity) → float
- volatility(returns) → float                      # annualized std

Statistical:
- statistical_significance(predictions, actuals) → dict  # binomial test p-value
- bootstrap_confidence_interval(returns, n=10000) → tuple  # 95% CI
- deflated_sharpe_ratio(sharpe, n_trials, n_obs, skew, kurtosis) → float  # Bailey & Lopez de Prado

Aggregate:
- compute_all_metrics(equity, returns, trades) → dict  # returns all above as a dict
```

2. Each function should handle edge cases (empty trades, zero division, etc.) gracefully
3. Use `scipy.stats` for statistical tests
4. Annualization factor: assume 6 bars/day × 252 days = 1512 bars/year for 4H data

**Outputs:**
- Fully implemented `utils/metrics.py` with all functions above
- Each function has a docstring explaining inputs, outputs, and formula

**Tests:**
- Test with known values: e.g., equity curve [100, 110, 105, 115] should produce correct max_drawdown
- Test edge cases: empty series, single trade, all wins, all losses
- Test statistical functions produce reasonable p-values
- `compute_all_metrics()` returns a dict with all expected keys

---

### Task 1.6: Backtesting Engine

**What:** Replace the 25-line backtest with a proper engine using vectorbt. Must support costs, slippage, long+short, position sizing, and trade-level logging.

**Where:**
- REWRITE: `pipeline/backtest.py`

**Inputs:** Read `research_algorithmic_trading.md` Section 6 (Risk Management) for context on what metrics to compute.

**Steps:**
1. Read current `pipeline/backtest.py` (25 lines)
2. Rewrite using `vectorbt` (already in requirements.txt) as the backend
3. Implement:

```python
class BacktestEngine:
    def __init__(self, initial_capital=100000, commission=0.0001, slippage=0.0001):
        """
        commission: as fraction of trade value (0.01% default for gold)
        slippage: as fraction of price (next-bar-open execution)
        """

    def run(self, prices, signals, confidence=None) -> BacktestResult:
        """
        prices: pd.Series of close prices
        signals: pd.Series of -1/0/+1 (short/flat/long)
        confidence: optional pd.Series 0-1 for position sizing
        Returns BacktestResult with equity curve, trades, metrics
        """

    def run_with_sizing(self, prices, signals, confidence, method='fixed_fractional', risk_per_trade=0.02):
        """Position sizing: fixed_fractional, volatility_scaled, or kelly"""

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame        # entry_time, exit_time, direction, entry_price, exit_price, pnl, duration
    positions: pd.Series        # position at each bar
    metrics: dict               # all metrics from utils/metrics.py
```

4. Support both long and short positions
5. Default execution model: enter at next bar's open (not current close)
6. Trade-level logging: every trade recorded with entry/exit price, time, direction, P&L, duration
7. Drawdown tracking: peak equity, current drawdown at each bar
8. Integration point: `metrics` dict should use `utils/metrics.py` functions (can import them or compute independently — the metrics module may not exist yet when this task runs, so implement basic versions inline if needed and note them for later replacement)

**Outputs:**
- `BacktestEngine` class with full functionality
- `BacktestResult` dataclass with equity curve, trades DataFrame, metrics dict
- Support for transaction costs, slippage, long+short, position sizing

**Tests:**
- Backtest with all-long signals on rising prices → positive P&L
- Backtest with zero signals → zero P&L minus any costs
- Verify trade count matches number of signal changes
- Verify transaction costs reduce P&L (same signals with/without costs)
- Verify slippage reduces P&L
- Trade DataFrame has correct entry/exit prices and durations

---

### Task 1.7: Clean Up Empty Stubs

**What:** Remove all empty stub files that have only 1 line (placeholder pass/import). These create confusion about what the project supports.

**Where:**
- DELETE or GUT: `models/ensemble.py`, `models/anomaly_detection.py`, `models/autoencoder.py`, `models/multitask_model.py`, `models/rl_agent.py`, `utils/metrics.py` (ONLY if Task 1.5 hasn't already replaced it), `utils/helpers.py`, `pipeline/trainer.py`

**Steps:**
1. Read each file. If it's only 1 line (stub), delete the file
2. Check all imports across the codebase — if any file imports from a deleted stub, remove that import
3. Check `main.py` for any references to deleted modules
4. Do NOT delete `utils/metrics.py` if it has been implemented by Task 1.5

**Outputs:**
- No stub files remain
- No broken imports
- `main.py` still runs without import errors

**Tests:**
- `python -c "import main"` succeeds (no import errors)
- No file in the project is a 1-line stub

---

# WAVE 2 — Data Quality & Labels (Depends on Wave 1)

> With clean data loading and fixed models, now fix what the models are predicting and what features they see.

---

### Task 2.1: First-Pass Feature Reduction

**What:** Remove the 61 CDL (candlestick pattern) columns and highly correlated feature pairs (>0.95) from the dataset. Target: reduce from 161 to ~40-50 features.

**Where:**
- MODIFY: `pipeline/data_loader.py` (add feature filtering logic)
- CREATE: `utils/feature_utils.py` (correlation filter, CDL filter)

**Depends on:** Task 1.1 (leaking columns removed)

**Steps:**
1. Read `Data/gold_4h.csv` column names
2. Identify all columns starting with `CDL_` — these are candlestick pattern indicators (very weak signal on 4H data). Create a list.
3. In `utils/feature_utils.py`, implement:
   - `drop_cdl_columns(df) → df` — drops all CDL_ prefixed columns
   - `drop_correlated_features(df, threshold=0.95) → df` — computes pairwise correlation matrix, for each pair above threshold, drops the one with lower variance
   - `get_feature_report(df) → dict` — returns count of features, list of dropped features, correlation statistics
4. Integrate into `data_loader.py` so feature reduction happens after loading, before normalization
5. Log which features were dropped and why

**Outputs:**
- `utils/feature_utils.py` with reusable feature filtering functions
- Updated `data_loader.py` that applies feature reduction
- Log output showing: original feature count, CDL features dropped, correlated features dropped, final count

**Tests:**
- No CDL_ columns remain after filtering
- No feature pair has correlation > 0.95 after filtering
- Feature count is between 30-60 (down from 161)
- Verify the remaining features include core indicators (RSI, ATR, SMA, EMA, MACD, volume)

---

### Task 2.2: Triple Barrier Labeling

**What:** Implement triple barrier labeling (Lopez de Prado) to replace the simple binary up/down target. Labels are path-dependent: take-profit barrier, stop-loss barrier, time expiry barrier.

**Where:**
- CREATE: `utils/labeling.py`
- MODIFY: `pipeline/data_loader.py` (add option to use triple barrier labels)
- MODIFY: `pipeline/config.py` (add labeling config)

**Depends on:** Task 1.1 (clean data)

**Reference:** `research_algorithmic_trading.md` Section 4 — Lopez de Prado's Key Contributions

**Steps:**
1. Implement in `utils/labeling.py`:

```python
def triple_barrier_labels(
    close: pd.Series,
    high: pd.Series,
    low: pd.Series,
    atr: pd.Series,
    tp_multiplier: float = 2.0,      # take-profit = tp_multiplier * ATR
    sl_multiplier: float = 1.0,      # stop-loss = sl_multiplier * ATR
    max_holding_period: int = 12,     # bars (12 × 4H = 48 hours)
) -> pd.DataFrame:
    """
    For each bar, look forward up to max_holding_period bars.
    - If price hits tp_multiplier * ATR above entry first → label = 1 (win)
    - If price hits sl_multiplier * ATR below entry first → label = -1 (loss)
    - If neither hit within max_holding_period → label = 0 (flat/timeout)

    Returns DataFrame with columns: label, barrier_hit ('tp', 'sl', 'time'), bars_to_hit
    """
```

2. Use HIGH and LOW prices (not just close) for barrier checks — this catches intrabar barrier hits
3. Barriers are volatility-scaled using ATR — they adapt to market conditions
4. Add to `CONFIG`:
   ```python
   'labeling': {
       'method': 'triple_barrier',  # or 'binary'
       'tp_multiplier': 2.0,
       'sl_multiplier': 1.0,
       'max_holding_period': 12,
   }
   ```
5. Modify `load_data()` to support both `binary` and `triple_barrier` labeling methods
6. Also implement `binary_labels(close, horizon=1)` for backward compatibility

**Outputs:**
- `utils/labeling.py` with `triple_barrier_labels()` and `binary_labels()`
- Config option to switch between labeling methods
- Updated data loader supporting both

**Tests:**
- On a known price sequence, verify labels are correct:
  - Price goes up 3×ATR within 5 bars → label=1, barrier_hit='tp', bars_to_hit=5
  - Price goes down 1.5×ATR within 3 bars → label=-1, barrier_hit='sl', bars_to_hit=3
  - Price stays within barriers for 12 bars → label=0, barrier_hit='time'
- Verify no look-ahead: label at bar i only uses data from bars i+1 to i+max_holding_period
- Verify label distribution is reasonable (not 95% one class)
- Verify ATR-scaling: wider barriers in volatile periods, tighter in calm periods

---

### Task 2.3: Fractional Differentiation

**What:** Implement fractional differentiation for price-based features to achieve stationarity while retaining memory. Use d ≈ 0.2 which retains >90% correlation with original series.

**Where:**
- CREATE: `utils/fracdiff.py`
- MODIFY: `pipeline/data_loader.py` (apply to price columns)

**Depends on:** Task 1.1 (clean data)

**Reference:** `research_algorithmic_trading.md` Section 4

**Steps:**
1. Install `fracdiff` library (already a known dependency) OR implement from scratch:
   ```python
   def frac_diff(series: pd.Series, d: float = 0.2, threshold: float = 1e-5) -> pd.Series:
       """
       Apply fractional differentiation of order d to a time series.
       d=0: original series (full memory, non-stationary)
       d=1: standard returns (stationary, no memory)
       d≈0.2: sweet spot — stationary with >90% memory retained
       threshold: minimum weight to include in the kernel (truncation)
       """
   ```
2. Implement `find_min_d(series, p_value=0.05) -> float` — finds the minimum d that makes the series stationary (ADF test p-value < threshold)
3. Apply fractional differentiation to price-based columns: `close`, `high`, `low`, `open` (and any derived price features like SMA, EMA values)
4. Do NOT apply to already-stationary features (RSI, returns, oscillators)
5. Add to data pipeline: after loading, identify price-level columns, apply fracdiff, replace originals

**Outputs:**
- `utils/fracdiff.py` with `frac_diff()` and `find_min_d()`
- Integrated into data loading pipeline
- Price columns are fractionally differenced before model training

**Tests:**
- ADF test on fracdiff'd close price → p-value < 0.05 (stationary)
- Correlation between fracdiff'd and original close > 0.9 (memory retained)
- Verify d=0 returns original series
- Verify d=1 returns standard returns (diff)
- No NaN explosion (proper threshold truncation)

---

### Task 2.4: CUSUM Event-Based Sampling

**What:** Implement CUSUM filter to identify structurally meaningful events. Instead of predicting every 4H bar (most are noise), only generate labels/predictions at CUSUM-triggered events.

**Where:**
- CREATE: `utils/sampling.py`
- MODIFY: `pipeline/data_loader.py` (add CUSUM sampling option)

**Depends on:** Task 1.1 (clean data)

**Reference:** `research_algorithmic_trading.md` Section 4 — Lopez de Prado methods

**Steps:**
1. Implement in `utils/sampling.py`:
   ```python
   def cusum_filter(close: pd.Series, threshold: float = None) -> pd.DatetimeIndex:
       """
       Symmetric CUSUM filter. Detects structural breaks in price.
       threshold: if None, use daily volatility (std of returns)
       Returns: DatetimeIndex of event timestamps where CUSUM triggered
       """
       # Track cumulative positive and negative deviations
       # When either exceeds threshold, emit an event and reset
   ```
2. The threshold should default to the rolling standard deviation of returns (adaptive)
3. Also implement `get_cusum_events(df, threshold=None) -> pd.DataFrame` that returns the subset of rows at CUSUM event points
4. Add config option: `'sampling': {'method': 'all', ...}` or `'sampling': {'method': 'cusum', 'threshold': None}`
5. Integrate: when CUSUM sampling is enabled, `load_data()` returns only event rows (but preserves full data for feature computation)

**Outputs:**
- `utils/sampling.py` with CUSUM filter implementation
- Config option for sampling method
- Integrated into data pipeline

**Tests:**
- On synthetic data with known structural breaks, verify CUSUM triggers at break points
- Verify CUSUM reduces dataset size (should keep 20-50% of bars, not 100%)
- Verify no look-ahead: CUSUM at time t only uses data up to time t
- Verify events are more frequent in volatile periods, less in calm periods

---

### Task 2.5: Benchmark Strategies

**What:** Implement 3 benchmark strategies that every model must beat to prove its worth.

**Where:**
- CREATE: `utils/benchmarks.py`

**Depends on:** Task 1.6 (backtesting engine)

**Steps:**
1. Implement three benchmarks:

```python
def buy_and_hold(prices: pd.Series) -> pd.Series:
    """Returns equity curve for buy-and-hold gold strategy"""

def random_strategy(prices: pd.Series, n_simulations: int = 10000, seed: int = 42) -> dict:
    """
    Monte Carlo random entry/exit.
    Returns: dict with 'mean_return', 'std_return', 'p95_return', 'p5_return', 'equity_curves'
    """

def sma_crossover(prices: pd.Series, fast: int = 50, slow: int = 200) -> pd.Series:
    """
    Simple moving average crossover.
    Long when fast > slow, flat otherwise.
    Returns signals series (-1/0/+1)
    """
```

2. Each benchmark should return results compatible with `BacktestEngine.run()` input or output format
3. Random strategy runs 10,000 simulations and returns distribution statistics

**Outputs:**
- `utils/benchmarks.py` with 3 benchmark implementations
- Each produces results that can be compared against model performance

**Tests:**
- Buy and hold on a rising price series → positive return
- Random strategy mean return is near zero (no edge)
- SMA crossover generates signals at correct crossover points
- All benchmarks produce valid equity curves (no NaN, monotonic when no trades)

---

### Task 2.6: SHAP Analysis Integration

**What:** Implement SHAP value extraction for XGBoost to understand which features carry predictive power.

**Where:**
- CREATE: `utils/shap_analysis.py`

**Depends on:** Task 1.1 (clean data), Task 1.2 (fixed normalization)

**Steps:**
1. Implement:
```python
def compute_shap_values(model, X_test, feature_names=None) -> dict:
    """
    Compute SHAP values for an XGBoost model.
    Returns: dict with 'shap_values', 'feature_importance' (sorted), 'top_n_features'
    """

def plot_shap_summary(shap_values, X_test, feature_names, save_path=None):
    """SHAP summary plot (beeswarm)"""

def plot_shap_importance(shap_values, feature_names, top_n=20, save_path=None):
    """Bar chart of mean |SHAP| values"""

def get_top_features(shap_values, feature_names, n=30) -> list:
    """Return top-N features by mean absolute SHAP value"""
```

2. Use the `shap` library (add to requirements if not present)
3. Handle both binary classification and multi-class SHAP values

**Outputs:**
- `utils/shap_analysis.py` with SHAP computation and visualization
- Functions that work with any XGBoost model

**Tests:**
- SHAP values have same shape as input features
- Feature importance is sorted descending
- Top-N features list has N items
- Plots save to disk without error

---

# WAVE 3 — Architecture & Validation (Depends on Wave 2)

> Build the pipeline architecture (standard interfaces) and validation framework. These two tracks can run in parallel.

---

## Track A: Pipeline Architecture

### Task 3A.1: TradingPipeline Interface & PipelineOutput

**What:** Define the core abstractions that ALL models will implement. This is the single most impactful architectural decision.

**Where:**
- CREATE: `shared/__init__.py`
- CREATE: `shared/interfaces.py`

**Reference:** `research_architecture.md` — Target Architecture, The Standard Interface

**Steps:**
1. Create the `shared/` directory
2. Define in `shared/interfaces.py`:

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd

@dataclass
class PipelineOutput:
    signals: pd.Series          # -1, 0, +1 (short, flat, long)
    confidence: pd.Series       # 0.0 to 1.0 (calibrated probability)
    metadata: dict = field(default_factory=dict)  # model name, version, params, training metrics

@dataclass
class BacktestResult:
    equity_curve: pd.Series
    returns: pd.Series
    trades: pd.DataFrame
    positions: pd.Series
    metrics: dict

class TradingPipeline(ABC):
    """Base interface for all trading pipelines."""

    @abstractmethod
    def train(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None) -> dict:
        """Train the pipeline. Returns training metrics dict."""
        ...

    @abstractmethod
    def predict(self, data: pd.DataFrame) -> PipelineOutput:
        """Generate trading signals with calibrated confidence."""
        ...

    @abstractmethod
    def evaluate(self, data: pd.DataFrame) -> dict:
        """Evaluate on data. Returns metrics dict."""
        ...

    @abstractmethod
    def save(self, path: str) -> None:
        """Persist model to disk."""
        ...

    @abstractmethod
    def load(self, path: str) -> None:
        """Load model from disk."""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """Pipeline identifier."""
        ...
```

3. Also create `shared/calibration.py`:
```python
def calibrate_probabilities(y_true, y_prob, method='isotonic') -> CalibratedClassifier:
    """
    Calibrate model probabilities using Platt scaling or isotonic regression.
    method: 'platt' (logistic) or 'isotonic'
    Returns fitted calibrator that can transform raw probabilities to calibrated ones.
    """

def apply_calibration(raw_probs, calibrator) -> np.ndarray:
    """Apply fitted calibrator to raw probability outputs."""
```

**Outputs:**
- `shared/interfaces.py` with `TradingPipeline`, `PipelineOutput`, `BacktestResult`
- `shared/calibration.py` with probability calibration
- Clear docstrings on every method

**Tests:**
- `PipelineOutput` can be instantiated with valid data
- A mock class implementing `TradingPipeline` must implement all abstract methods or raise TypeError
- Calibration: on perfectly calibrated probs, calibration should be near-identity

---

### Task 3A.2: Shared Infrastructure Layer

**What:** Build the shared infrastructure that all pipelines will use: data loader, feature store, metrics, backtest.

**Where:**
- CREATE: `shared/data_loader.py` (new, centralized — different from `pipeline/data_loader.py`)
- CREATE: `shared/feature_store.py`
- MOVE/WRAP: `utils/metrics.py` → `shared/metrics.py` (or re-export)
- MOVE/WRAP: `pipeline/backtest.py` → `shared/backtest.py` (or re-export)

**Depends on:** Task 3A.1 (interfaces), Task 1.5 (metrics), Task 1.6 (backtest engine)

**Steps:**
1. Create `shared/data_loader.py`:
   - Single function `load_gold_data(config) -> dict` that returns `{'train': df, 'val': df, 'test': df, 'scaler': scaler}`
   - Handles: loading CSV, excluding leaking columns, feature reduction, normalization (train-only fit), splitting
   - Applies CUSUM sampling if configured
   - Applies triple barrier labeling if configured
   - This is the ONE place all data loading happens

2. Create `shared/feature_store.py`:
   ```python
   class FeatureStore:
       def __init__(self, cache_dir="data/features/"):
           ...
       def get_features(self, data, feature_set_name, version) -> pd.DataFrame:
           """Compute features, cache as Parquet. Return cached if exists."""
       def invalidate(self, feature_set_name=None):
           """Clear cache for a feature set or all."""
       def list_feature_sets(self) -> list:
           """List available cached feature sets."""
   ```

3. Re-export metrics and backtest from shared:
   - `shared/metrics.py` can import from `utils/metrics.py`
   - `shared/backtest.py` can import from `pipeline/backtest.py`

**Outputs:**
- `shared/data_loader.py` — centralized data loading
- `shared/feature_store.py` — Parquet-based feature cache
- `shared/metrics.py` and `shared/backtest.py` — re-exports

**Tests:**
- `load_gold_data()` returns dict with train/val/test DataFrames
- Train/val/test have no overlapping indices
- Feature store: compute, cache, load — verify cached result equals computed result
- Feature store: invalidate clears the cache

---

### Task 3A.3: Configuration System

**What:** Replace the flat `CONFIG` dict with a hierarchical, YAML-based configuration system.

**Where:**
- CREATE: `shared/config.py`
- CREATE: `configs/default.yaml`
- CREATE: `configs/pipelines/gradient_boosting.yaml`
- MODIFY: `pipeline/config.py` (deprecate, point to new system)

**Steps:**
1. Use Pydantic (v2) for validation. Define config models:

```python
class DataConfig(BaseModel):
    data_path: Path = Path("Data/gold_4h.csv")
    target_type: str = "classification"
    target_horizon: int = 1
    test_size: float = 0.2
    val_size: float = 0.1
    exclude_cols: list[str] = ["target", "date", "future_returns", "signal"]
    labeling: LabelingConfig = LabelingConfig()
    sampling: SamplingConfig = SamplingConfig()

class LabelingConfig(BaseModel):
    method: str = "triple_barrier"
    tp_multiplier: float = 2.0
    sl_multiplier: float = 1.0
    max_holding_period: int = 12

class PipelineConfig(BaseModel):
    name: str
    model_type: str
    params: dict = {}
    features: list[str] | None = None  # None = use all

class CombinerConfig(BaseModel):
    method: str = "weighted_voting"
    confidence_threshold: float = 0.6

class AppConfig(BaseModel):
    data: DataConfig = DataConfig()
    pipelines: list[PipelineConfig] = []
    combiner: CombinerConfig = CombinerConfig()
    random_state: int = 42
    environment: str = "dev"  # dev/test/production

def load_config(path: str = "configs/default.yaml") -> AppConfig:
    """Load config from YAML, validate with Pydantic."""
```

2. Create `configs/default.yaml` with sensible defaults
3. Support CLI overrides: `main.py --config configs/experiment_1.yaml`

**Outputs:**
- Pydantic-based config system with YAML loading
- Default config file
- Example pipeline config

**Tests:**
- Load default config → all fields have valid values
- Override a field in YAML → override takes effect
- Invalid config (e.g., test_size=2.0) → validation error

---

### Task 3A.4: Wrap Existing Models as Pipelines

**What:** Wrap each existing model into the `TradingPipeline` interface. Organize into pipeline directories.

**Where:**
- CREATE: `pipelines/__init__.py`
- CREATE: `pipelines/gradient_boosting/__init__.py`
- CREATE: `pipelines/gradient_boosting/xgboost_pipeline.py`
- CREATE: `pipelines/deep_learning/__init__.py`
- CREATE: `pipelines/deep_learning/lstm_pipeline.py`
- CREATE: `pipelines/deep_learning/tcn_pipeline.py`
- CREATE: `pipelines/deep_learning/transformer_pipeline.py`
- CREATE: `pipelines/deep_learning/gru_pipeline.py`
- CREATE: `pipelines/deep_learning/cnn_lstm_pipeline.py`
- CREATE: `pipelines/deep_learning/crnn_pipeline.py`
- CREATE: `pipelines/deep_learning/tabnet_pipeline.py`

**Depends on:** Task 3A.1 (interfaces), Task 1.3 (fixed sequence models)

**Steps:**
1. For each existing model, create a pipeline wrapper class:
```python
class XGBoostPipeline(TradingPipeline):
    def __init__(self, config: dict):
        self.config = config
        self.model = None
        self.calibrator = None

    def train(self, train_data, val_data=None):
        # Use existing baseline_ml.py logic
        # After training, fit calibrator on validation set
        ...

    def predict(self, data) -> PipelineOutput:
        raw_probs = self.model.predict_proba(X)[:, 1]
        calibrated = apply_calibration(raw_probs, self.calibrator)
        signals = pd.Series(np.where(calibrated > 0.5, 1, -1))
        return PipelineOutput(signals=signals, confidence=pd.Series(calibrated), metadata={...})
```
2. Each pipeline wraps the EXISTING model code — don't rewrite the models, just wrap them
3. Every pipeline must output calibrated probabilities (use `shared/calibration.py`)
4. Keep the old model files as-is for backward compatibility — the pipeline wrappers import from them

**Outputs:**
- Every existing model has a pipeline wrapper implementing `TradingPipeline`
- All pipelines output `PipelineOutput` with calibrated confidence
- Organized in `pipelines/` directory structure

**Tests:**
- Each pipeline can `train()` on sample data without error
- Each pipeline's `predict()` returns valid `PipelineOutput`
- Confidence values are between 0 and 1
- Signals are -1, 0, or +1

---

### Task 3A.5: Orchestrator & Main Entry Point

**What:** Rewrite `main.py` to use the new pipeline architecture. Sequential orchestrator that runs all pipelines and collects results.

**Where:**
- REWRITE: `main.py`

**Depends on:** Task 3A.1-3A.4

**Steps:**
1. New `main.py` should:
   - Parse CLI args (--config, --pipeline, --run-all)
   - Load config
   - Set seeds
   - Load data via `shared/data_loader.py`
   - Instantiate pipelines based on config
   - Train each pipeline
   - Collect `PipelineOutput` from each
   - (Future: feed to combiner — for now, just collect and compare)
   - Evaluate each pipeline via `shared/metrics.py`
   - Log results
   - Save results

2. Support running a single pipeline or all:
   ```
   python main.py --pipeline gradient_boosting
   python main.py --run-all
   python main.py --config configs/experiment_v2.yaml
   ```

**Outputs:**
- New `main.py` that orchestrates pipelines via standard interface
- CLI argument support
- Results comparison output

**Tests:**
- `python main.py --pipeline gradient_boosting` runs XGBoost pipeline end-to-end
- `python main.py --run-all` runs all configured pipelines
- Each pipeline produces metrics output
- No crashes, clean shutdown

---

## Track B: Validation Framework (Parallel with Track A)

### Task 3B.1: Walk-Forward Validation

**What:** Implement walk-forward validation with rolling and expanding windows.

**Where:**
- CREATE: `shared/validation.py`

**Depends on:** Task 1.5 (metrics), Task 1.6 (backtest engine)

**Steps:**
1. Implement:
```python
class WalkForwardValidator:
    def __init__(self, n_splits=5, train_size=None, test_size=None, expanding=False, gap=0):
        """
        n_splits: number of walk-forward windows
        train_size: fixed training window size (bars). None = expanding
        test_size: test window size (bars)
        expanding: if True, training window grows; if False, rolls forward
        gap: embargo gap between train and test (bars) to prevent leakage
        """

    def split(self, X) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate train/test index arrays for each split."""

    def validate(self, pipeline: TradingPipeline, data: pd.DataFrame, prices: pd.Series) -> ValidationResult:
        """
        Run full walk-forward validation.
        For each split: train on train window, predict on test window, evaluate.
        Returns ValidationResult with per-split metrics and aggregate statistics.
        """

@dataclass
class ValidationResult:
    per_split_metrics: list[dict]    # metrics for each walk-forward window
    aggregate: dict                   # mean, std, min, max of each metric
    walk_forward_efficiency: float    # OOS return / IS return
    equity_curves: list[pd.Series]   # per-split equity curves
```

2. Walk-Forward Efficiency = mean(OOS returns) / mean(IS returns) — values >0.7 indicate good transferability
3. Respect chronological order — no future data leakage between splits
4. Support embargo gap (purging) between train and test windows

**Outputs:**
- `WalkForwardValidator` class with rolling and expanding modes
- `ValidationResult` with comprehensive metrics

**Tests:**
- Splits are chronologically ordered (all train indices < all test indices per split)
- No overlap between train and test in any split
- Gap between train end and test start equals the embargo parameter
- Works with both rolling and expanding windows
- Aggregate metrics have correct mean/std

---

### Task 3B.2: Combinatorial Purged Cross-Validation (CPCV)

**What:** Implement CPCV (Lopez de Prado) — the gold standard for financial ML validation.

**Where:**
- ADD TO: `shared/validation.py`

**Depends on:** Task 3B.1

**Reference:** `research_algorithmic_trading.md` Section 5

**Steps:**
1. Implement:
```python
class CPCValidator:
    def __init__(self, n_groups=6, n_test_groups=2, purge_gap=6, embargo_gap=12):
        """
        n_groups: number of chronological groups to split data into
        n_test_groups: number of groups in each test set
        purge_gap: bars to remove at train/test boundary (prevent leakage)
        embargo_gap: bars to skip after purge (additional safety)
        """

    def split(self, X) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate all valid train/test combinations."""

    def validate(self, pipeline, data, prices) -> CPCVResult:
        """Run all CPCV paths and aggregate."""

    def probability_of_backtest_overfitting(self) -> float:
        """PBO: fraction of CPCV paths where IS performance > OOS performance. Target: < 0.5"""
```

2. Can use `mlfinlab` if available, or implement from scratch
3. The key insight: CPCV generates C(n_groups, n_test_groups) paths, giving a distribution of performance
4. PBO < 0.5 means the strategy is more likely to be genuine than overfit

**Outputs:**
- `CPCValidator` class
- PBO computation
- Distribution of OOS performance across all paths

**Tests:**
- Number of paths equals C(n_groups, n_test_groups)
- All train/test splits are valid (no overlap after purging)
- PBO is between 0 and 1
- On a random strategy, PBO should be near 0.5

---

### Task 3B.3: Monte Carlo Robustness Testing

**What:** Implement Monte Carlo simulation to test strategy robustness.

**Where:**
- ADD TO: `shared/validation.py`

**Steps:**
1. Implement:
```python
class MonteCarloValidator:
    def __init__(self, n_simulations=10000, seed=42):
        ...

    def test_trade_shuffle(self, trades: pd.DataFrame) -> dict:
        """Shuffle trade order, recompute equity curves. Tests if returns depend on ordering."""

    def test_parameter_sensitivity(self, pipeline, data, param_ranges: dict) -> dict:
        """Randomly perturb parameters within ranges, retrain, evaluate. Tests robustness to params."""

    def test_skip_trades(self, trades: pd.DataFrame, skip_rate=0.1) -> dict:
        """Randomly skip X% of trades, recompute metrics. Simulates missed signals."""

    def test_bootstrap_returns(self, returns: pd.Series) -> dict:
        """Bootstrap resample returns to generate confidence intervals."""

    def full_robustness_report(self, ...) -> dict:
        """Run all tests, return comprehensive report."""
```

**Outputs:**
- `MonteCarloValidator` with 4 robustness tests
- Each returns distribution statistics (mean, std, percentiles)
- Full robustness report combining all tests

**Tests:**
- 10,000 shuffled equity curves have reasonable spread
- Bootstrap CI contains the observed metric value
- Skip-trade test shows graceful degradation (not catastrophic failure)

---

# WAVE 4 — Features & New Models (Depends on Wave 3)

> With the architecture in place, add new features and models. These tasks are highly parallelizable.

---

### Task 4.1: Volume-Based Feature Engineering

**What:** Implement OBV, MFI, A/D, VWAP, and volume profile features from existing OHLCV data.

**Where:**
- CREATE: `shared/features/volume.py`

**Reference:** `research_order_flow.md` — Tier 1 (Volume Indicators) and Tier 5 (Volume Profile)

**Steps:**
1. Implement all volume indicators from `research_order_flow.md` Tier 1:
   - `compute_obv(close, volume)` → Series
   - `compute_mfi(high, low, close, volume, period=14)` → Series
   - `compute_ad(high, low, close, volume)` → Series
   - `compute_vwap(high, low, close, volume)` → Series
   - `compute_obv_slope(obv, period=4)` → Series
   - `detect_obv_divergence(close, obv, period=14)` → Series (binary flag)
   - `compute_ad_slope(ad, period=4)` → Series
2. Volume profile features:
   - `compute_volume_profile(close, volume, period=20)` → DataFrame with POC, VA_high, VA_low
   - `compute_distance_from_poc(close, poc)` → Series
   - `compute_above_value_area(close, va_high)` → Series (binary)

**Outputs:**
- All volume indicator functions with clear docstrings
- Each returns a pd.Series aligned with input index

**Tests:**
- OBV is cumulative and increases on up-moves
- MFI is bounded [0, 100]
- Functions handle NaN gracefully (first few rows)
- Output length matches input length

---

### Task 4.2: Multi-Timeframe Data Acquisition & Alignment

**What:** Set up data loading for 1H, Daily, and Weekly gold data with proper alignment to prevent look-ahead bias.

**Where:**
- REWRITE: `utils/multi_timeframe_utils.py`
- MODIFY: `shared/data_loader.py`

**Reference:** `research_multi_timeframe.md` — Timeframe Alignment section

**Steps:**
1. Rewrite `utils/multi_timeframe_utils.py`:
   ```python
   def align_timeframes(
       df_4h: pd.DataFrame,
       df_daily: pd.DataFrame = None,
       df_weekly: pd.DataFrame = None,
       df_1h: pd.DataFrame = None,
   ) -> pd.DataFrame:
       """
       Merge multi-timeframe data using merge_asof with backward direction.
       Higher-TF features are SHIFTED by 1 period before merge.
       This ensures a 4H bar only sees COMPLETED higher-TF bars.
       """
       # Shift higher-TF by one period
       if df_daily is not None:
           df_daily = df_daily.shift(1)
           df_4h = pd.merge_asof(df_4h, df_daily, left_index=True, right_index=True, direction='backward')
       # Same for weekly, 1H aggregation
   ```

2. Implement 1H aggregation per 4H bar:
   ```python
   def aggregate_1h_to_4h(df_1h: pd.DataFrame) -> pd.DataFrame:
       """Compute 1H-derived features aggregated per 4H bar window."""
       # session_indicator, intra_bar_trend, intra_bar_volatility, london_open_momentum
   ```

3. Implement multi-TF feature extraction:
   ```python
   def compute_daily_features(df_daily: pd.DataFrame) -> pd.DataFrame:
       """6 features: daily_trend, daily_atr, daily_rsi, daily_ma_distance, daily_close_vs_open, daily_range_position"""

   def compute_weekly_features(df_weekly: pd.DataFrame) -> pd.DataFrame:
       """3 features: weekly_trend, weekly_ma_position, weekly_atr"""

   def compute_1h_features(df_1h: pd.DataFrame) -> pd.DataFrame:
       """5 features: session_indicator, intra_bar_trend, intra_bar_volatility, london_open_momentum, volume_profile"""
   ```

4. Standardize all timestamps to UTC
5. Drop weekend bars, don't forward-fill across gaps

**Outputs:**
- `align_timeframes()` with `merge_asof` + shift (no look-ahead bias)
- Feature extraction functions for each timeframe
- 1H aggregation per 4H bar
- All timestamps in UTC

**Tests:**
- A 4H bar at 08:00 Tuesday does NOT see Tuesday's daily close
- A 4H bar on Monday does NOT see this week's weekly close
- `merge_asof` direction is 'backward' everywhere
- All higher-TF features are shifted by 1 period before merge
- No NaN rows from merge (backward fill from last available)
- Weekend/holiday bars are dropped

---

### Task 4.3: External Data Integration

**What:** Build data loaders for external data sources: DXY, Treasury yields, VIX, COT data, COMEX open interest.

**Where:**
- CREATE: `shared/features/external.py`
- CREATE: `shared/data_sources/` directory with per-source loaders

**Reference:** `research_order_flow.md` (COT, open interest), roadmap Phase 5.5

**Steps:**
1. Implement data loaders (prioritize free sources):
   ```python
   # shared/data_sources/macro.py
   def load_dxy(start, end) -> pd.DataFrame  # via yfinance or FRED
   def load_treasury_yields(start, end) -> pd.DataFrame  # 2Y, 10Y from FRED
   def load_vix(start, end) -> pd.DataFrame  # via yfinance

   # shared/data_sources/cot.py
   def load_cot_gold(start, end) -> pd.DataFrame  # from CFTC
   def compute_cot_features(cot_data) -> pd.DataFrame
       # cot_net_speculative, cot_net_change, cot_extreme, cot_commercial_hedge_ratio

   # shared/data_sources/comex.py
   def load_open_interest(start, end) -> pd.DataFrame  # from CME
   def compute_oi_features(oi_data, prices) -> pd.DataFrame
       # oi_change, oi_price_confirm, oi_divergence
   ```

2. Each loader should cache downloaded data locally (Parquet)
3. All data aligned to UTC
4. Merge into 4H data using `merge_asof(direction='backward')` — same alignment rules as multi-TF

**Outputs:**
- Data loaders for DXY, yields, VIX, COT, open interest
- Feature computation functions for each source
- Local caching to avoid repeated downloads
- Proper temporal alignment

**Tests:**
- Each loader returns a DataFrame with datetime index
- COT features: cot_extreme is binary, cot_net_change is the delta
- OI features: oi_price_confirm follows the interpretation matrix from research_order_flow.md
- No future data leakage in alignment

---

### Task 4.4: LightGBM & CatBoost Pipelines

**What:** Add LightGBM and CatBoost as direct comparisons to XGBoost. Both already in requirements.txt.

**Where:**
- CREATE: `pipelines/gradient_boosting/lightgbm_pipeline.py`
- CREATE: `pipelines/gradient_boosting/catboost_pipeline.py`

**Depends on:** Task 3A.1 (TradingPipeline interface)

**Steps:**
1. Each implements `TradingPipeline` interface
2. LightGBM: use GOSS boosting, comparable hyperparams to XGBoost
3. CatBoost: enable native categorical feature handling for day-of-week, session, regime
4. Both must output calibrated probabilities
5. Default hyperparams comparable to XGBoost config (300 estimators, lr 0.03, depth 5)

**Outputs:**
- Two new pipeline classes, fully implementing `TradingPipeline`
- Calibrated probability output
- Config files for each

**Tests:**
- Both can train and predict without errors
- Output is valid `PipelineOutput`
- Confidence is calibrated (values between 0-1, roughly matching observed rates)

---

### Task 4.5: HMM Regime Detection

**What:** Implement 3-state Hidden Markov Model for market regime classification (bull/quiet, choppy/uncertain, bear/crisis).

**Where:**
- CREATE: `pipelines/regime/hmm_pipeline.py`

**Reference:** Roadmap Phase 6.1

**Steps:**
1. Use `hmmlearn` library (add to requirements if not present)
2. Implement:
```python
class HMMRegimePipeline(TradingPipeline):
    """
    3-state HMM on returns + volatility.
    Outputs regime labels (0=bull, 1=choppy, 2=bear) + regime probability as confidence.
    Does NOT output direct trading signals — instead outputs regime state
    that other pipelines and the combiner use.
    """
    def train(self, train_data, val_data=None):
        # Fit GaussianHMM with n_components=3 on [returns, volatility]
        ...

    def predict(self, data) -> PipelineOutput:
        # Predict most likely regime sequence
        # Signal: 1 if bull, -1 if bear, 0 if choppy
        # Confidence: posterior probability of the predicted regime
        ...
```

3. Features for HMM: returns, rolling volatility (ATR or realized vol), volume change
4. Regime labels should be auto-assigned based on mean return in each state (highest mean = bull, lowest = bear)
5. Also provide `get_regime_features(data) -> pd.DataFrame` that returns regime as a feature column for other models

**Outputs:**
- `HMMRegimePipeline` implementing `TradingPipeline`
- Regime classification (3 states)
- Regime probability output
- `get_regime_features()` for other pipelines to use

**Tests:**
- HMM converges (check log-likelihood increases during training)
- 3 distinct states with different mean returns
- Regime labels are somewhat persistent (not flipping every bar)
- Works on the gold dataset without numerical issues

---

### Task 4.6: Temporal Fusion Transformer (TFT)

**What:** Implement TFT — the highest-priority new deep learning architecture. Interpretable attention, variable selection, handles mixed inputs.

**Where:**
- CREATE: `pipelines/deep_learning/tft_pipeline.py`

**Reference:** `research_algorithmic_trading.md` Section 2 — TFT description

**Steps:**
1. Use `pytorch-forecasting` library (add to requirements) which has TFT implementation, OR implement a simplified version using raw PyTorch
2. If using pytorch-forecasting:
   - Create TimeSeriesDataSet from the gold data
   - Configure known future inputs (calendar features) vs unknown future inputs
   - Use TemporalFusionTransformer class
3. Wrap in `TradingPipeline` interface
4. Extract and log: variable importance from attention weights, temporal attention patterns
5. Hyperparams: hidden_size=64, attention_head_size=4, dropout=0.1, learning_rate=0.001

**Outputs:**
- `TFTPipeline` implementing `TradingPipeline`
- Calibrated probability output
- Interpretability: variable importance extraction

**Tests:**
- Model trains without errors on gold data
- Output is valid `PipelineOutput`
- Variable importance can be extracted
- Predictions are not constant (model is actually learning)

---

# WAVE 5 — Combiner & Ensemble (Depends on Wave 4)

> With multiple pipelines producing signals, build the parent algorithm that combines them.

---

### Task 5.1: Weighted Voting Combiner (Level 1)

**What:** Build the Level 1 combiner — weight each pipeline by its rolling out-of-sample accuracy.

**Where:**
- CREATE: `combiner/__init__.py`
- CREATE: `combiner/weighted_voting.py`

**Reference:** `research_architecture.md` — The Combiner (6 Levels)

**Steps:**
1. Implement:
```python
class WeightedVotingCombiner:
    def __init__(self, lookback=100, min_weight=0.05):
        """
        lookback: rolling window for computing pipeline accuracy
        min_weight: minimum weight for any pipeline (prevents zero-weight)
        """

    def combine(self, pipeline_outputs: dict[str, PipelineOutput], recent_actuals: pd.Series = None) -> PipelineOutput:
        """
        Combine multiple pipeline outputs into one.
        Weights = rolling OOS accuracy of each pipeline.
        Output signal = weighted vote. Confidence = weighted average confidence.
        """

    def get_weights(self) -> dict[str, float]:
        """Current pipeline weights."""

    def compute_signal_correlation(self, pipeline_outputs: dict[str, PipelineOutput]) -> pd.DataFrame:
        """Pairwise correlation between pipeline signals — for independence tracking."""
```

2. Signal independence tracking: log pairwise correlation and downweight correlated signals
3. Disagreement handling: if pipelines split 50/50, reduce confidence toward 0.5

**Outputs:**
- `WeightedVotingCombiner` that combines any number of pipeline outputs
- Rolling weight computation
- Signal correlation tracking

**Tests:**
- With 3 pipelines all saying "buy", combined signal is "buy" with high confidence
- With 2 buy / 1 sell, combined signal is "buy" with reduced confidence
- Weights sum to 1.0
- Better-performing pipelines get higher weights

---

### Task 5.2: Stacking Meta-Learner (Level 2)

**What:** XGBoost trained on pipeline probability outputs + market features to learn when each pipeline is reliable.

**Where:**
- CREATE: `combiner/stacking.py`

**Depends on:** Task 5.1, Task 3A.4 (pipelines producing outputs)

**Steps:**
1. Implement:
```python
class StackingCombiner:
    def __init__(self):
        self.meta_model = None  # XGBoost

    def train(self, pipeline_outputs: dict[str, PipelineOutput], market_features: pd.DataFrame, actuals: pd.Series):
        """
        CRITICAL: Train ONLY on out-of-sample predictions from child pipelines.
        Features = [pipeline1_prob, pipeline2_prob, ..., market_features]
        Target = actual outcome
        """

    def combine(self, pipeline_outputs, market_features) -> PipelineOutput:
        """Generate combined signal from meta-model."""
```

2. **CRITICAL:** The meta-model must only see OOS predictions from children. Use nested CV or walk-forward:
   - Split data into K folds
   - For each fold: train children on other folds, predict on this fold
   - Collect all OOS predictions → train meta-model on these

**Outputs:**
- `StackingCombiner` with OOS-only meta-model training
- XGBoost meta-learner that learns pipeline reliability

**Tests:**
- Meta-model never sees in-sample predictions
- Meta-model outperforms simple weighted voting on validation data (or at least matches)
- SHAP on meta-model shows which pipelines get most weight

---

### Task 5.3: Confidence Thresholding & Risk Management

**What:** Only trade when combined confidence exceeds a threshold. Implement position sizing based on confidence.

**Where:**
- CREATE: `combiner/risk.py`

**Reference:** Roadmap Phase 7.1, Phase 8

**Steps:**
1. Implement:
```python
class RiskManager:
    def __init__(self, confidence_threshold=0.6, max_position=1.0, risk_per_trade=0.02):
        ...

    def apply(self, combined_output: PipelineOutput, atr: pd.Series, equity: float) -> pd.DataFrame:
        """
        Returns DataFrame with: signal, position_size, stop_loss, take_profit
        - If confidence < threshold: signal = 0 (no trade)
        - Position size = confidence * max_position (scaled by ATR for vol targeting)
        - Stop loss = entry - sl_multiplier * ATR
        - Take profit = entry + tp_multiplier * ATR
        """

    def volatility_target(self, atr, target_vol=0.10):
        """Scale position inversely to ATR"""

    def half_kelly(self, win_rate, avg_win_loss_ratio):
        """Kelly criterion at 50%"""

    def drawdown_check(self, equity_curve, max_drawdown=0.15) -> bool:
        """Returns True if drawdown exceeds threshold (circuit breaker)"""
```

**Outputs:**
- `RiskManager` with confidence filtering, position sizing, drawdown circuit breaker
- Vol targeting and Kelly sizing
- Trade-level stop/take-profit computation

**Tests:**
- Low confidence signals are filtered out (position_size = 0)
- Higher confidence → larger position size
- Drawdown circuit breaker triggers at correct threshold
- Kelly sizing gives reasonable fractions (not > 100%)

---

### Task 5.4: MLflow Integration

**What:** Set up MLflow experiment tracking for all pipelines and the combiner.

**Where:**
- CREATE: `shared/tracking.py`
- MODIFY: pipeline wrappers to log to MLflow

**Steps:**
1. Implement:
```python
class ExperimentTracker:
    def __init__(self, tracking_uri="mlruns", experiment_name="ben10"):
        mlflow.set_tracking_uri(tracking_uri)

    def log_pipeline_run(self, pipeline_name, params, metrics, model=None):
        """Log a pipeline training run to MLflow."""

    def log_combiner_run(self, combiner_type, pipeline_weights, combined_metrics):
        """Log a combiner run comparing pipeline outputs."""

    def get_best_run(self, pipeline_name, metric="sharpe_ratio") -> dict:
        """Get the best run for a pipeline by metric."""

    def register_model(self, pipeline_name, run_id, stage="staging"):
        """Register model in MLflow Model Registry."""
```

2. Each pipeline's `train()` should automatically log: params, all metrics, model artifact
3. `mlflow ui` should show all experiments

**Outputs:**
- `ExperimentTracker` wrapping MLflow
- Integration points in pipeline wrappers
- Model registry support

**Tests:**
- After a training run, `mlflow ui` shows the experiment
- Metrics, params, and model artifact are logged
- Model can be loaded from registry

---

# WAVE 6 — Integration & Testing (Depends on Wave 5)

> Wire everything together and test end-to-end.

---

### Task 6.1: End-to-End Integration Test

**What:** Full pipeline run: load data → train all models → combine signals → backtest → validate → report.

**Where:**
- CREATE: `tests/test_e2e.py`
- MODIFY: `main.py` (ensure it orchestrates the full flow)

**Steps:**
1. Write an end-to-end test that:
   - Loads gold_4h.csv with all fixes (leakage removed, normalization fixed, features reduced)
   - Trains at least XGBoost + one DL model
   - Gets `PipelineOutput` from each
   - Runs weighted voting combiner
   - Backtests the combined signal
   - Runs walk-forward validation
   - Produces a metrics report
2. Verify: no accuracy is 100% (leakage is gone), no backtest return is -100%
3. Verify: combined signal outperforms random baseline

**Outputs:**
- `tests/test_e2e.py` that validates the entire pipeline
- Documented expected ranges for key metrics

**Tests:**
- XGBoost accuracy is between 45-65% (not 100%)
- Backtest returns are reasonable (not -100%)
- Combiner produces valid output
- Walk-forward validation completes without error

---

### Task 6.2: Dashboard Update

**What:** Update the Streamlit dashboard to work with the new pipeline architecture.

**Where:**
- MODIFY: `dashboard/` (all files)

**Steps:**
1. Update training page to use `TradingPipeline` interface
2. Add combiner visualization (pipeline weights, signal agreement)
3. Add validation results display (walk-forward windows, PBO)
4. Add regime visualization (HMM states overlay on price chart)
5. Fix pages to work with actual available data files

**Outputs:**
- Updated dashboard working with new architecture
- Combiner and validation visualization
- No broken pages

---

# DEPENDENCY GRAPH SUMMARY

```
WAVE 1 (all parallel — no dependencies):
  1.1 Remove data leakage
  1.2 Fix normalization
  1.3 Fix sequence models
  1.4 Seed management
  1.5 Trading metrics module
  1.6 Backtesting engine
  1.7 Clean up stubs

WAVE 2 (depends on Wave 1):
  2.1 Feature reduction         ← 1.1
  2.2 Triple barrier labeling   ← 1.1
  2.3 Fractional differentiation ← 1.1
  2.4 CUSUM sampling            ← 1.1
  2.5 Benchmark strategies      ← 1.6
  2.6 SHAP analysis             ← 1.1, 1.2

WAVE 3 (depends on Wave 2):
  Track A (Architecture):
    3A.1 TradingPipeline interface
    3A.2 Shared infrastructure    ← 3A.1, 1.5, 1.6
    3A.3 Configuration system
    3A.4 Wrap models as pipelines ← 3A.1, 1.3
    3A.5 Orchestrator             ← 3A.1-3A.4
  Track B (Validation — parallel with Track A):
    3B.1 Walk-forward validation  ← 1.5, 1.6
    3B.2 CPCV                     ← 3B.1
    3B.3 Monte Carlo testing      ← 3B.1

WAVE 4 (depends on Wave 3):
  4.1 Volume features             (parallel)
  4.2 Multi-TF alignment          (parallel)
  4.3 External data integration   (parallel)
  4.4 LightGBM + CatBoost        ← 3A.1
  4.5 HMM regime detection       ← 3A.1
  4.6 TFT model                  ← 3A.1

WAVE 5 (depends on Wave 4):
  5.1 Weighted voting combiner   ← 3A.4
  5.2 Stacking meta-learner      ← 5.1
  5.3 Risk management            ← 5.1
  5.4 MLflow integration          (parallel)

WAVE 6 (depends on Wave 5):
  6.1 End-to-end integration test
  6.2 Dashboard update
```

---

# AGENT ASSIGNMENT GUIDE

For maximum parallelism, assign agents as follows:

**Wave 1 — 7 agents simultaneously:**
| Agent | Task | Estimated Scope |
|-------|------|----------------|
| Agent A | 1.1 Data leakage | Small — config + data_loader changes |
| Agent B | 1.2 Normalization fix | Small — data_loader rewrite |
| Agent C | 1.3 Sequence model fix | Medium — new utility + 4 model files |
| Agent D | 1.4 Seed management | Small — new utility + main.py |
| Agent E | 1.5 Trading metrics | Medium — full metrics module |
| Agent F | 1.6 Backtesting engine | Large — full backtest rewrite |
| Agent G | 1.7 Stub cleanup | Small — delete files, fix imports |

**Wave 2 — 6 agents simultaneously:**
| Agent | Task | Estimated Scope |
|-------|------|----------------|
| Agent A | 2.1 Feature reduction | Medium |
| Agent B | 2.2 Triple barrier labeling | Large |
| Agent C | 2.3 Fractional differentiation | Medium |
| Agent D | 2.4 CUSUM sampling | Medium |
| Agent E | 2.5 Benchmarks | Small |
| Agent F | 2.6 SHAP analysis | Medium |

**Wave 3 — 5+ agents (two tracks):**
| Agent | Task | Estimated Scope |
|-------|------|----------------|
| Agent A | 3A.1 Interface + 3A.3 Config | Medium |
| Agent B | 3A.2 Shared infrastructure | Medium |
| Agent C | 3A.4 Wrap models | Large |
| Agent D | 3B.1 Walk-forward | Large |
| Agent E | 3B.2 CPCV + 3B.3 Monte Carlo | Large |

**Wave 4 — 6 agents:**
| Agent | Task | Estimated Scope |
|-------|------|----------------|
| Agent A | 4.1 Volume features | Medium |
| Agent B | 4.2 Multi-TF alignment | Large |
| Agent C | 4.3 External data | Large |
| Agent D | 4.4 LightGBM + CatBoost | Medium |
| Agent E | 4.5 HMM regime | Medium |
| Agent F | 4.6 TFT | Large |

**Wave 5 — 4 agents:**
| Agent | Task | Estimated Scope |
|-------|------|----------------|
| Agent A | 5.1 Weighted voting | Medium |
| Agent B | 5.2 Stacking | Large |
| Agent C | 5.3 Risk management | Medium |
| Agent D | 5.4 MLflow | Medium |

**Wave 6 — 2 agents:**
| Agent | Task | Estimated Scope |
|-------|------|----------------|
| Agent A | 6.1 E2E test | Large |
| Agent B | 6.2 Dashboard | Medium |

---

# FILE CONFLICT AVOIDANCE

Tasks within the same wave should NOT modify the same files. If two tasks need to modify the same file, one should CREATE a new file and the other should MODIFY the existing one — merge after both complete.

**Known shared files (coordinate carefully):**
- `pipeline/data_loader.py` — touched by 1.1, 1.2, 2.1, 2.2, 2.3, 2.4. Solution: 1.1 and 1.2 modify different functions. Wave 2 tasks create utilities in `utils/` and the integration into data_loader is done by Task 3A.2 (shared data loader).
- `main.py` — touched by 1.4, 3A.5. Solution: 1.4 adds one line (seed call). 3A.5 rewrites the file (later wave, no conflict).
- `pipeline/config.py` — touched by 1.1, 2.2. Solution: 1.1 adds to exclude_cols. 2.2 adds labeling config. Different dict keys, no conflict.

---

*This plan covers Phases 0-8 of the roadmap. Phases 9-11 (live trading, production, multi-asset) are future work that depends on the research results from this implementation.*
