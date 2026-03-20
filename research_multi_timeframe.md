# Multi-Timeframe Analysis — Research Report

**Date:** 2026-03-19
**Purpose:** Inform multi-timeframe strategy for the Ben10 gold trading pipeline

---

## Decision: Use 4 Timeframes (1H / 4H / Daily / Weekly)

### Why These Four

The 4:1 ratio rule (Alexander Elder, 1986) provides enough separation for genuinely different market structure while maintaining signal coherence:

| Timeframe | Role | Features to Extract |
|-----------|------|-------------------|
| **Weekly** | Macro context | 2-3 features: trend direction, MA position, weekly ATR |
| **Daily** | Trend filter | 5-8 features: trend, ATR, RSI, MA distance, candle type, range position |
| **4H** | Primary prediction | Core feature set — all ML models predict here |
| **1H** | Entry timing + session dynamics | 3-5 features: session indicator, intra-bar trend/volatility, London momentum |

### Why Not Sub-Hourly

Research evidence against sub-hourly data for 4H prediction:

- **Signal-to-noise ratio:** Bouchaud et al. (2025) show sub-30-minute data is dominated by microstructure noise and mean reversion — no directional signal for ML.
- **Overfitting:** IEEE study found random forests on minute-level data showed "superior backtest performance that did not translate to better out-of-sample results."
- **Practical threshold:** ~4-6x finer than prediction horizon is the limit. For 4H, that's 1H (4:1). Going to 15min (16:1) is marginal; 1min (240:1) is counterproductive.
- **Diminishing returns:** Going from 1→2 TFs: large improvement. 2→3: meaningful. 3→4+: marginal at best, significant overfitting risk.

### Why Not More Than 4

- Each timeframe adds features but training samples stay the same (determined by 4H prediction horizon)
- More features = more spurious correlations for tree-based models to exploit
- Debugging becomes harder — can't tell which TF contributes signal vs noise
- Research consensus: "less is more" for ML in finance

---

## Timeframe Alignment — Preventing Look-Ahead Bias

**This is the #1 pitfall in multi-timeframe ML.** The most common source of unrealistically good backtests.

### The Problem

A 4H bar closing at 08:00 UTC on Tuesday must NOT see:
- Tuesday's daily close (not yet available — closes at ~21:00 UTC)
- This week's weekly close (not yet available — closes Friday)
- Any indicator computed on incomplete higher-TF bars

### The Fix

```python
# WRONG: outer join — may align 4H bar with same-day daily bar (not yet closed)
merged = df_4h.join(df_daily, how='outer').dropna()

# RIGHT: shift higher-TF by one period, then merge_asof backward
daily_features = daily_features.shift(1)   # Use yesterday's completed daily
weekly_features = weekly_features.shift(1)  # Use last week's completed weekly

merged = pd.merge_asof(
    df_4h, daily_features,
    left_index=True, right_index=True,
    direction='backward'  # Only see data from the past
)
```

### Additional Alignment Rules

- **Standardize on UTC.** Gold 4H bar boundaries vary by broker (some use EST, some GMT+2). Pick UTC and stick to it.
- **Handle weekends/holidays.** Drop weekend bars. Don't forward-fill across gaps — it creates artificial continuity.
- **Resample carefully.** Pandas `resample()` can create empty bars and misalign boundaries. Use manual OHLC with explicit `first/max/min/last/sum`.
- **4H bar DST issue.** Depending on whether bars start at 00:00 or 01:00 UTC (due to DST), you get different candle patterns. Pick one convention.

---

## Multi-Timeframe Model Architectures

### Approach A: Selective Higher-TF Features (Simplest — Start Here)

Add 10-15 features from Daily/Weekly/1H to the 4H feature set. Train all models on this enriched set.

**Pros:** Simple, works with all existing models, no architectural changes.
**Cons:** No explicit modeling of TF hierarchy.
**Expected improvement:** 3-8% in prediction quality.

### Approach B: Per-Timeframe Models + Stacking Meta-Learner

Train independent models per timeframe. Feed their probability outputs (not binary 0/1) plus cross-TF features into a meta-model.

**Pros:** Models specialize per TF. Stacking captures cross-TF interactions.
**Cons:** More complex, need enough data per TF.
**Expected improvement over A:** 2-5%.
**Research:** Stacking ensembles achieve 5-15% performance improvement over simple voting (CIKM 2021).

### Approach C: Hierarchical (Most Sophisticated)

Top-down knowledge flow: Daily model → 4H model → 1H model.
Higher-TF model outputs become input features for lower-TF models.

**Pros:** Matches how institutional traders think (top-down). Prevents noise propagation upward.
**Cons:** Most complex to implement and debug.
**Research basis:** Multi-agent RL framework (Expert Systems with Applications, 2023).

### Approach D: Cross-Timeframe Attention (State of the Art)

MSTAN (Multi-Scale Temporal Attention Network): 2D reconstruction of multi-scale representations with temporal hybrid attention.

**Pros:** Jointly captures short-term fluctuations and long-term trends.
**Cons:** Complex, research-stage, less battle-tested.
**When to consider:** After Approaches A-C are validated.

---

## Gold-Specific Session Patterns

Gold has uniquely strong session-dependent behavior — this is why 1H data adds genuine value:

| Session | Hours (UTC) | Behavior | % of Daily Range |
|---------|-------------|----------|-----------------|
| Asian | 00:00–08:00 | Low vol, range-bound, mean-reverting | ~15-20% |
| London Open | 08:00–13:00 | Volatility spike, often sets daily direction | ~35-40% |
| London-NY Overlap | 13:00–16:00 | Peak liquidity, largest directional moves, tightest spreads | ~30-35% |
| NY Afternoon | 16:00–21:00 | Declining vol, often retraces earlier moves | ~10-15% |

**Key events:**
- LBMA AM Fix (10:30 London): Major institutional price discovery
- LBMA PM Fix (15:00 London): Most liquid fix, largest directional moves
- COMEX open: Price gap from London close

**Features to extract:**
- `session`: categorical (Asian/London/Overlap/NY)
- `london_open_breakout`: did price break Asian session range in first London hour?
- `session_volatility_ratio`: ATR(London) / ATR(Asian) — detects unusual activity
- `time_of_day`: cyclical encoding (sin/cos of hour)

---

## Storage and Compute — Non-Issues

For a single instrument (gold), storage is trivial:

| Timeframe | Size / 20 Years |
|-----------|----------------|
| 1-minute | ~320 MB |
| 1-hour | ~5.4 MB |
| 4-hour | ~1.4 MB |
| Daily | ~220 KB |
| Weekly | ~46 KB |

Adding 1H + Daily + Weekly to your existing 4H data adds ~6 MB. Compute impact on XGBoost: seconds, not hours. On deep learning: roughly linear with number of additional feature channels.

---

## Key Sources

- Bouchaud et al. (2025), "Trends and Reversion in Financial Markets on Time Scales from Minutes to Decades" — https://arxiv.org/html/2501.16772v1
- SSRN (2025), "Fractal Structures in Financial Markets: Cross-Asset and Multi-Timeframe Pattern Similarity"
- IEEE (2021), "Exploring the Use of Data at Multiple Granularity Levels in ML-Based Stock Trading"
- CIKM (2021), "Stock Trend Prediction with Multi-Granularity Data"
- MSTAN (2025), "Multi-Scale Temporal Attention Network for Stock Prediction"
- Expert Systems with Applications (2023), "Multi-Agent Deep RL Framework for Algorithmic Trading"
- Alexander Elder, "Trading for a Living" — Triple Screen system
- Multi-Timeframe Ensemble-HMM Voting Framework (2025)

---

*This document should be revisited if the primary timeframe changes or if the project expands to assets with different session characteristics.*
