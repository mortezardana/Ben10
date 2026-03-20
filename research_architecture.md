# System Architecture — Research Report

**Date:** 2026-03-19
**Purpose:** Inform the architectural evolution of Ben10 from monolithic script to modular pipeline system

---

## Decision: Modular Monolith with Independent Pipelines

Not microservices. Not a distributed system. A single codebase with clear boundaries, independent research pipelines, shared infrastructure, and a parent combiner.

This is the pattern used by every major quant firm — Citadel's "pod shop" model, Two Sigma's platform approach, Jane Street's functional monolith. Infrastructure is centralized; research is independent.

---

## Why Not Microservices

| Factor | Microservices | Modular Monolith |
|--------|--------------|-------------------|
| Latency | Network overhead between services | In-process function calls |
| Consistency | Distributed state is hard | Shared memory is simple |
| Debugging | Distributed tracing needed | Standard debugger works |
| Deployment | Per-service CI/CD | Single deploy |
| Team threshold | Benefits at 10+ developers | Works at any scale |
| Data sharing | API calls, serialization overhead | Shared DataFrame in memory |

For a solo developer trading gold on 4H candles: all pipelines share the same data, run in Python, and execute sequentially or with basic parallelism. Microservices would add complexity with zero benefit.

**When to reconsider:** If you scale to a team of 5+ with genuinely independent deployment needs, or if live trading requires sub-second signal combination from multiple data sources running on different schedules.

---

## The Alpha Factory Pattern

WorldQuant's approach: 4 million independent "alphas" (weak predictive signals) combined into a "mega-alpha." Key principles:

- **Individual alphas are cheap and disposable.** Many are simple formulaic expressions.
- **Low pairwise correlation is the goal.** WorldQuant's average: 15.9%. Diversity is the edge.
- **No single alpha needs to be profitable.** The aggregate signal is what trades.
- **Internal crossing reduces costs.** Buy signals from one alpha offset sell signals from another.

For Ben10, this translates to: each pipeline is an "alpha factory" producing signals. The combiner is the "mega-alpha" that trades.

---

## Target Architecture

```
Ben10/
├── shared/                          # Centralized infrastructure
│   ├── data_loader.py               # Single source of truth for data
│   ├── feature_store.py             # Compute once, cache as Parquet
│   ├── metrics.py                   # Standardized evaluation
│   └── backtest.py                  # Common backtesting engine
│
├── pipelines/                       # Independent research pods
│   ├── gradient_boosting/           # XGBoost, LightGBM, CatBoost
│   │   ├── config.yaml
│   │   ├── features.py
│   │   ├── train.py
│   │   └── predict.py
│   ├── deep_learning/               # LSTM, Transformer, TFT, TCN
│   │   └── ...
│   ├── regime/                      # HMM, clustering, anomaly detection
│   │   └── ...
│   └── experimental/                # RL, foundation models
│       └── ...
│
├── combiner/                        # Parent algorithm
│   ├── meta_model.py                # Stacking / meta-labeling
│   ├── portfolio.py                 # Position sizing
│   └── router.py                    # Regime-conditional routing
│
└── main.py                          # Orchestrator
```

### The Standard Interface

```python
@dataclass
class PipelineOutput:
    signals: pd.Series          # -1, 0, +1
    confidence: pd.Series       # 0.0 to 1.0
    metadata: dict              # model name, version, params

class TradingPipeline(ABC):
    def train(self, train_data: pd.DataFrame) -> None: ...
    def predict(self, data: pd.DataFrame) -> PipelineOutput: ...
    def evaluate(self, data: pd.DataFrame) -> dict: ...
```

Rules:
- Each pipeline has its own config and hyperparameters
- No pipeline accesses another pipeline's internal state
- Shared infrastructure is read-only for pipelines
- Communication only through `PipelineOutput` objects

---

## The Combiner (6 Levels)

### Level 1 — Weighted Voting
Weight each pipeline by rolling out-of-sample accuracy. Simplest, good starting point.

### Level 2 — Stacking Meta-Learner
XGBoost trained on pipeline probability outputs + market features. Learns when each pipeline is reliable.

### Level 3 — Meta-Labeling (Lopez de Prado)
Best pipeline picks direction. Secondary model decides whether to act. Probability = position size.

### Level 4 — Regime-Conditional Routing
HMM regime state determines pipeline weights. Explicit, interpretable.

### Level 5 — Mixture of Experts
Gating network learns dynamic routing based on market features.

### Level 6 — Full Alpha Factory
Hundreds of weak signals combined via portfolio optimization. WorldQuant scale. Future aspiration.

### Disagreement Handling
When pipelines disagree:
1. **Conservative:** Sit out entirely. Often best risk-adjusted approach.
2. **Proportional:** Trade at reduced size (2/3 agree = 66% size).
3. **Learned:** Meta-model trained with disagreement as a feature.

---

## Communication & Parallelism

### During Research (Phases 0-7)

**Sequential (start here):**
```python
results = {}
for name, pipeline in pipelines.items():
    results[name] = pipeline.run(data)
combined = combiner.combine(results)
```

**Ray (when training is slow):**
```python
import ray

@ray.remote
def run_pipeline(pipeline, data):
    return pipeline.train_and_predict(data)

futures = [run_pipeline.remote(p, data) for p in pipelines]
results = ray.get(futures)
```

Ray's shared-memory object store loads data once; all pipelines access it.

### During Live Trading (Phases 9-10)

**Docker Compose (minimal):**
- App container (Python, all pipelines)
- MLflow container (experiment tracking)
- TimescaleDB container (time-series storage)
- Redis (only if real-time signal pub/sub needed)

### Skip Entirely
- Kafka (not processing millions of events/sec)
- Kubernetes (not scaling horizontally)
- gRPC (no distributed services)
- Celery (overkill for batch pipelines)
- Multi-repo (one person, one codebase)

---

## Feature Store

### Phase 4-7: Simple Parquet Cache

```python
class FeatureStore:
    def __init__(self, cache_dir="data/features/"):
        self.cache_dir = cache_dir

    def get_features(self, data, feature_set_name, version):
        cache_path = f"{self.cache_dir}/{feature_set_name}_v{version}.parquet"
        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path)
        features = self._compute_features(data, feature_set_name)
        features.to_parquet(cache_path)
        return features
```

### Phase 9-10: Feast
When you need online serving (real-time feature computation for live inference), graduate to Feast. It provides offline store (for training) and online store (for inference) with the same feature definitions.

---

## Experiment Tracking

**MLflow** is the right choice:
- Lightweight: `pip install mlflow`, `mlflow ui` for dashboard
- Each pipeline logs to its own MLflow experiment
- Model Registry with staging/production states per pipeline
- Combiner logs to a separate experiment comparing pipeline outputs
- Comparison across pipeline iterations

---

## Reference Architectures

### NautilusTrader
Modular monolith with event bus. Rust core, Python control plane. Single-threaded deterministic event loop. Strategies communicate via MessageBus, never directly. Same code runs in backtest and live.

### QuantConnect LEAN
Algorithm Framework with pluggable modules: Universe Selection → Alpha Generation → Portfolio Construction → Execution → Risk Management. Each module independently replaceable.

### Freqtrade
Python-native strategy framework. Strategies inherit from `IStrategy`. FreqAI integration for adaptive ML prediction.

### WorldQuant BRAIN
Alpha factory platform. Millions of formulaic alphas. Low pairwise correlation. Combined via portfolio optimization.

---

## Key Sources

- Citadel pod structure: https://navnoorbawa.substack.com/p/how-millennium-citadel-and-point72
- Two Sigma engineering: https://www.twosigma.com/articles/building-for-the-future-architecture-month-at-two-sigma/
- Jane Street technology (OCaml monolith): https://www.janestreet.com/technology/
- WorldQuant 101 Formulaic Alphas: https://arxiv.org/pdf/1601.00991
- NautilusTrader architecture: https://nautilustrader.io/docs/latest/concepts/architecture/
- Monolith vs Microservices (ByteByteGo): https://blog.bytebytego.com/p/monolith-vs-microservices-vs-modular
- Quant trading system architecture: https://mbrenndoerfer.com/writing/quant-trading-system-architecture-infrastructure
- MIGA Mixture of Experts for stocks: https://arxiv.org/html/2410.02241v1
- Goldman Sachs combining signals: https://www.gsam.com/content/dam/gsam/pdfs/institutions/en/articles/2018/Combining_Investment_Signals_in_LongShort_Strategies.pdf
- MBATS open-source reference: https://github.com/saeed349/Microservices-Based-Algorithmic-Trading-System
- MLflow: https://mlflow.org/
- Ray: https://www.ray.io/
- Feast feature store: https://feast.dev/

---

*Revisit this when scaling to multiple team members or when live trading requires real-time multi-source signal combination.*
