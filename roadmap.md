# 🗺️ Development Roadmap for Gold Trading AI Pipeline

This roadmap outlines the step-by-step stages for developing, training, evaluating, and deploying the AI-based gold trading system using a combination of machine learning, deep learning, and reinforcement learning models.

---

## 📌 Phase 1: Project Setup
- [x] Define goals and scope
- [x] Create folder structure
- [x] Add README.md and roadmap.md
- [x] Create requirements.txt

---

## 📦 Phase 2: Data Preparation
- [ ] Load and clean 4H gold OHLCV dataset (2004–2025)
- [ ] Add all TA-Lib indicators
- [ ] Normalize features (MinMax or Z-Score)
- [ ] Create target labels (binary classification, regression)
- [ ] Split dataset (train, validation, test)

---

## 🤖 Phase 3: Individual Model Development

### Supervised Learning Models
- [ ] `baseline_ml.py` – XGBoost / LightGBM
- [ ] `lstm_model.py` – LSTM (Keras/TensorFlow)
- [ ] `cnn_lstm.py` – CNN + LSTM hybrid
- [ ] `tcn_model.py` – Temporal Convolutional Network
- [ ] `transformer_model.py` – Time Series Transformer

### Clustering & Anomaly
- [ ] `autoencoder.py` – Autoencoder + KMeans for regime detection
- [ ] `anomaly_detection.py` – Isolation Forest / One-Class SVM

### Multi-task and Reinforcement Learning
- [ ] `multitask_model.py` – Joint price + volatility model
- [ ] `rl_agent.py` – PPO / DQN with FinRL or Stable-Baselines3

---

## 🔗 Phase 4: Ensemble Techniques

- [ ] Voting / Weighted Average
- [ ] Stacking with Meta Model
- [ ] Multi-Stage Regime-Based Models
- [ ] Hybrid RL using ML signals as input

---

## 🧪 Phase 5: Evaluation & Backtesting

- [ ] Implement evaluator.py for metrics (accuracy, Sharpe, drawdown)
- [ ] Create backtest.py using `Backtrader` or `vectorbt`
- [ ] Visualize strategy equity curve, drawdowns, and trades

---

## ⚙️ Phase 6: Pipeline Orchestration

- [ ] `main.py` to train/evaluate/run all models
- [ ] `trainer.py` to loop through model training
- [ ] Config system in `config.py`

---

## 📈 Phase 7: Optimization & Tuning

- [ ] Feature selection (SHAP, Lasso, PCA)
- [ ] Hyperparameter tuning (Optuna / GridSearch)
- [ ] Add other timeframes (1h, daily)

---

## 📤 Phase 8: Reporting & Deployment

- [ ] Generate strategy reports (PDF, HTML)
- [ ] Export models and metrics
- [ ] Optionally deploy as a streamlit dashboard

---

## 🔄 Ongoing Improvements

- [ ] Monitor live data stream
- [ ] Update model weights periodically
- [ ] Add alerts or notifications for trade signals
