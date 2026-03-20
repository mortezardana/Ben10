# Ben10 — First Assessment

**Date:** 2026-03-19
**Assessed by:** Claude (Opus 4.6)
**Scope:** Full codebase review — architecture, ML correctness, trading viability, production readiness

---

## 1. Project Summary

Ben10 is a modular ML pipeline for predicting Gold (XAU/USD) direction on 4-hour candles. It currently trains 8 models (XGBoost, LSTM, CNN-LSTM, CRNN, GRU, TCN, Transformer, TabNet), runs a simple backtest on each, and visualizes results. A Streamlit dashboard provides interactive training and model comparison.

The dataset spans 2004–2025 with 100+ pre-computed technical indicators. All models perform binary classification: will the next candle close higher or lower?

---

## 2. What Works Well

- **Modular file structure.** One model per file, clean separation between pipeline, models, and utils.
- **Breadth of models.** 8 distinct architectures gives a solid foundation for comparison.
- **Multi-timeframe support.** The signal voting and per-timeframe model training are good ideas and a real differentiator.
- **Dashboard.** Having an interactive UI for training and comparison is valuable for rapid experimentation.
- **Logging.** Timestamped logs for every run.

---

## 3. Critical Issues

These are problems that compromise the validity of results. They must be fixed before any conclusions about model performance can be trusted.

### 3.1 Data Leakage in Normalization

**Location:** `main.py:25-26`, `pipeline/data_loader.py`

The scaler is fitted on the entire dataset before train/test splitting. This means test data statistics (mean, std) influence training data normalization. In production, you don't have access to future data.

**Impact:** Inflated accuracy — models appear better than they actually are.

**Fix:** Fit scaler on training split only. Transform val/test using the training scaler. Persist the scaler alongside the model for inference.

### 3.2 Broken Sequence Model Inputs

**Location:** `models/lstm_model.py`, `models/gru_model.py`, `models/crnn_model.py`, `models/cnn_lstm.py`

These models reshape data as `(samples, features, 1)` — each sample is a single timestep with features treated as a sequence. This defeats the purpose of recurrent/convolutional architectures. They're effectively operating as dense networks.

Only TCN and Transformer correctly create lookback windows of `(samples, timesteps, features)`.

**Impact:** These 4 models are not learning temporal patterns. Their results are meaningless as sequence models.

**Fix:** Implement proper sequence windowing (like TCN/Transformer already do) for all recurrent and convolutional models. Use a shared utility function for consistency.

### 3.3 Potential Indicator Leakage

The 100+ technical indicators in `gold_4h.csv` are pre-computed externally. If any indicator was calculated using the full dataset (e.g., a normalized indicator, or one that uses future bars in its smoothing), it introduces look-ahead bias.

**Impact:** Impossible to know without auditing the indicator computation. Could range from negligible to catastrophic.

**Fix:** Either recompute all indicators inside the pipeline (guaranteeing causal computation), or audit and document the external computation to confirm no leakage.

### 3.4 Single Train/Test Split

One static 70/10/20 split tells you how the model performs on one specific market period. Gold has distinct regimes — trending (2008–2011), ranging (2013–2018), volatile (2020–2022). A model that appears profitable on 2022–2025 may blow up on 2015–2018.

**Impact:** No confidence that results generalize across market conditions.

**Fix:** Implement walk-forward validation with multiple train/test windows across the full dataset. Report performance distribution, not a single number.

---

## 4. Backtesting Deficiencies

The current backtest (`pipeline/backtest.py`) is a 24-line function that calculates cumulative log returns based on shifted predictions. It is not suitable for evaluating a trading strategy.

### Missing components:

| Component | Current State | Impact |
|-----------|--------------|--------|
| Transaction costs | None | Overstates returns. Gold spread is ~0.3 pips; on 4H this adds up. |
| Slippage | None | Assumes perfect fill at close price. Unrealistic. |
| Short selling | Not implemented | Model predicts down (0) but doesn't act on it. Half the signals are wasted. |
| Position sizing | Always 100% | No risk management. A real strategy never goes all-in. |
| Stop-loss / take-profit | None | No risk bounds per trade. |
| Drawdown tracking | Not computed | Can't assess risk without knowing peak-to-trough decline. |
| Trade-level analysis | None | Can't inspect individual trades — only see aggregate equity curve. |

### Recommendation:

Build a proper backtesting engine or leverage `vectorbt` / `backtrader` (already in requirements.txt). These handle all of the above out of the box and are battle-tested.

---

## 5. Evaluation Gaps

### 5.1 Accuracy Is Not Enough

Binary accuracy tells you how often the model is right, but not whether it makes money. A model with 55% accuracy that's right on small moves and wrong on large moves will lose money. A model with 48% accuracy that catches big moves can be highly profitable.

### 5.2 Missing Trading Metrics

The following should be computed for every model:

- **Profit factor** — gross profit / gross loss
- **Max drawdown** — largest peak-to-trough decline in equity
- **Calmar ratio** — annualized return / max drawdown
- **Sortino ratio** — return / downside deviation (better than Sharpe for asymmetric returns)
- **Win rate** — % of profitable trades
- **Average win / average loss** — ratio of mean win size to mean loss size
- **Expectancy** — (win_rate * avg_win) - (loss_rate * avg_loss) — expected profit per trade
- **Trade count** — a model that generates 3 trades in 20 years is useless regardless of accuracy
- **Maximum consecutive losses** — stress test for psychological and capital resilience

### 5.3 No Statistical Significance

Is 52% accuracy on 10,000 predictions statistically significant? It depends. Without a significance test (e.g., binomial test, bootstrap confidence intervals, or permutation test), you can't distinguish skill from luck.

### 5.4 No Benchmark Comparison

Results should be compared against:
- **Buy and hold** — does the model beat simply holding gold?
- **Random baseline** — does it beat random entry/exit?
- **Moving average crossover** — does it beat the simplest technical strategy?

---

## 6. Architecture & Engineering

### 6.1 Pipeline Orchestration

`main.py` currently hardcodes the training of each model sequentially with inline logic. This doesn't scale.

**Needed:**
- A `trainer.py` that accepts any model through a common interface
- A model registry — register models by name, instantiate from config
- A run configuration (YAML or dataclass) that specifies which models to train, with what hyperparameters, on what data
- Separation of concerns: `main.py` should only parse CLI args and invoke the pipeline

### 6.2 Experiment Tracking

189+ log directories with no structured way to compare runs. You can't answer basic questions like "which model had the best Sharpe ratio last week?" without manually digging through files.

**Options (increasing complexity):**
- SQLite database logging metrics per run (lightweight, no infrastructure)
- MLflow (local server, good UI, model versioning)
- Weights & Biases (cloud-hosted, best visualization, free tier)

### 6.3 Configuration Management

Hyperparameters are scattered across model files. Each model has its own hardcoded values (epochs, batch size, learning rate, architecture choices). Changing anything requires editing source code.

**Fix:** Centralize all hyperparameters in config. Use a hierarchical config system (YAML + dataclass or Pydantic) so each model has its own section but shares common settings.

### 6.4 Reproducibility

No seed management beyond `random_state=42` in config. TensorFlow, PyTorch, NumPy, and Python's random module all need their seeds set for reproducible results. Add a `set_all_seeds(seed)` utility.

### 6.5 Empty Stubs

These files exist but are empty: `autoencoder.py`, `anomaly_detection.py`, `multitask_model.py`, `rl_agent.py`, `ensemble.py`, `metrics.py`, `helpers.py`, `trainer.py`.

Either implement them or remove them. Dead stubs create confusion about what the project actually supports.

---

## 7. Feature Engineering

### 7.1 Feature Selection

100+ features fed to every model is too many, especially for neural networks. Many technical indicators are highly correlated (SMA_20, EMA_20, DEMA_20 all track the same thing).

**Approaches to implement:**
- Correlation matrix filtering (drop features with >0.95 pairwise correlation)
- XGBoost feature importance ranking (use top-N)
- SHAP values for interpretable feature selection
- Recursive feature elimination
- PCA or autoencoder for dimensionality reduction

### 7.2 Target Engineering

Binary up/down is the simplest target but may not be the best:

- **3-class classification:** Up (>threshold), Flat, Down (<threshold) — filters out noise near zero
- **Regression + threshold:** Predict return magnitude, then apply a confidence threshold to trade
- **Volatility-adjusted target:** Define "up" relative to recent ATR, not absolute price change
- **Multi-horizon targets:** Predict 1-bar, 4-bar, and 12-bar direction — different models may excel at different horizons

### 7.3 External Features

Gold doesn't move in isolation. Consider adding:
- **USD Index (DXY)** — inverse correlation with gold
- **US Treasury yields** — opportunity cost of holding gold
- **S&P 500 / VIX** — risk appetite indicators
- **Oil prices** — commodity correlation
- **Fed funds rate / CPI data** — macro drivers
- **COT report data** — institutional positioning

These require additional data pipelines but could significantly improve predictive power.

---

## 8. Model-Specific Observations

### XGBoost (baseline)
- Strongest model in the current setup due to correct data handling
- 300 trees, depth 5, lr 0.03 are reasonable defaults
- Should add SHAP analysis for interpretability
- Consider LightGBM as a direct comparison — often faster with similar accuracy

### LSTM / GRU / CRNN / CNN-LSTM
- All broken due to incorrect input reshaping (see Section 3.2)
- Once fixed, these should use proper lookback windows (32–64 bars for 4H data)
- Add bidirectional variants as comparison
- Consider attention mechanisms on top of LSTM/GRU

### TCN
- Correctly implemented with sequence windowing
- Dilated causal convolutions are well-suited for financial time series
- Consider experimenting with different dilation rates and filter sizes

### Transformer
- Correctly implemented with multi-head attention
- 2 heads and 64 head size may be too small — experiment with 4–8 heads
- Add positional encoding if not already present (financial data has strong positional meaning)
- Consider a lighter variant (e.g., Informer, Autoformer) for efficiency

### TabNet
- Interesting choice — attention-based feature selection is useful for understanding which indicators matter
- Extract and log the feature attention masks — this gives free interpretability
- 200 epochs with patience 20 is reasonable

---

## 9. What's Missing Entirely

| Component | Why It Matters |
|-----------|---------------|
| **Regime detection** | Markets cycle between trending, ranging, and volatile states. One model doesn't fit all regimes. |
| **Ensemble methods** | Individual model predictions are noisy. Combining them properly is where real edge lives. |
| **Risk management** | No position sizing, no drawdown limits, no risk-per-trade. Can't go to production without this. |
| **Live data pipeline** | No way to feed real-time data for inference. Needed for paper trading and eventually live trading. |
| **Model monitoring** | No way to detect when a model's performance degrades over time (concept drift). |
| **Order execution layer** | No broker integration. Needed for live trading. |

---

## 10. Strengths to Build On

Despite the issues, this project has strong fundamentals:

1. **The modular structure is correct.** The separation of models, pipeline, and utils is the right architecture. It just needs to be filled in properly.
2. **Breadth of approaches.** Having 8+ model types is genuinely valuable for research. Most trading ML projects use one or two.
3. **Multi-timeframe thinking.** The signal voting system is a good idea and relatively uncommon in similar projects.
4. **Dashboard.** Having a UI accelerates experimentation enormously.
5. **20 years of data.** Enough to test across multiple market regimes with proper walk-forward validation.

The foundation is solid. The critical issues are fixable. The path from here to a credible research project — and potentially a production system — is clear.

---

*This assessment will be updated as the project evolves. See `roadmap.md` for the development plan.*
