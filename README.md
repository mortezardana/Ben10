# 🪙 Gold Trading AI Pipeline

This project started as my Data Science module coursework at the University of Roehampton, and quickly grew into its own repository and project because of the complexity and advancements compared to the coursework requirement. The name of the repo, Ben10, was given for the initial 10 models that were included.

This project builds a modular and extensible machine learning pipeline to predict and trade Gold (XAU/USD) based on 4-hour OHLCV data and technical indicators.

It includes **10 modeling approaches**, 4 **ensemble techniques**, and supports backtesting using realistic trading strategies.

---

## 📁 Project Structure

```
project_root/
├── data/               # Raw and processed datasets
├── models/             # Model implementations (ML, DL, RL, etc.)
├── pipeline/           # Training, evaluation, backtesting logic
├── notebooks/          # Exploratory notebooks
├── utils/              # Common utilities (metrics, logging, etc.)
├── main.py             # CLI runner to execute the pipeline
└── requirements.txt    # Python dependencies
```

---

## ✅ Features

- 🧠 10 Trading Models:
  - XGBoost, LSTM, TCN, Transformers, Autoencoders, RL Agents & more
- 🔀 Ensemble Learning:
  - Voting, stacking, multi-stage pipelines, hybrid RL strategies
- 📊 Technical Indicators:
  - 161+ indicators via TA-Lib
- 📅 Timeline:
  - 4H candle data from 2004–2025 (expandable to other timeframes)
- 📈 Backtesting:
  - Full backtest support via `Backtrader` or `vectorbt`
- 🔬 Evaluation:
  - Accuracy, Sharpe ratio, drawdown, confusion matrix, and more

---

## 🚀 Getting Started

1. Clone the repo:

```bash
git clone https://github.com/yourusername/gold-trading-ai.git
cd gold-trading-ai
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place your dataset in the `data/` folder (e.g., `gold_4h.csv`)

4. Run training pipeline:

```bash
python main.py --train
```

---

## ⚙️ Configuration

Modify `pipeline/config.py` to set:
- Training parameters
- Model hyperparameters
- Dataset paths
- Target horizon and prediction type

---

## 📈 Backtesting & Strategy

Once models are trained:
- Use `pipeline/backtest.py` to simulate trades based on predictions
- Apply thresholds, stop-loss, position sizing logic

---

## 🛠️ Models Included

| Model Type     | Technique                      |
|----------------|--------------------------------|
| ML             | XGBoost, LightGBM              |
| DL             | LSTM, TCN, Transformer, CNN-LSTM |
| Clustering     | Autoencoder + KMeans           |
| Anomaly        | Isolation Forest               |
| Multi-task     | Joint price + volatility models|
| RL             | PPO, DQN via FinRL             |
| Ensemble       | Voting, Stacking, Hybrid RL    |

---

## 📬 Contributions

PRs and ideas welcome! Let’s make this the most robust commodity trading AI pipeline out there.

---

## 📜 License

MIT License
