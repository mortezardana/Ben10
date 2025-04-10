# pipeline/evaluator.py

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from utils.logger import get_logger
from pathlib import Path

logger = get_logger("evaluator")


def _save_plot(path_or_str, filename):
    path = Path(path_or_str)
    path.mkdir(parents=True, exist_ok=True)
    plt.savefig(path / filename)


def plot_equity_curve(df, title="Equity Curve", plot_dir=None, model_name="model"):
    logger.info("Plotting equity curve")
    plt.figure(figsize=(12, 6))
    plt.plot(df['equity_curve'], label='Strategy Equity')
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if plot_dir:
        _save_plot(plot_dir, f"{model_name}_equity_curve.png")
    plt.show()


def plot_predictions(df, target_col='target', pred_col='prediction', title="Predictions vs Target", plot_dir=None,
                     model_name="model"):
    logger.info("Plotting predictions vs actual targets")
    plt.figure(figsize=(12, 4))
    plt.plot(df[target_col].values, label='Actual', alpha=0.7)
    plt.plot(df[pred_col].values, label='Predicted', alpha=0.7)
    plt.title(title)
    plt.xlabel("Index")
    plt.ylabel("Direction")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    if plot_dir:
        _save_plot(plot_dir, f"{model_name}_predictions_vs_actual.png")
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, plot_dir=None, model_name="model"):
    logger.info("Plotting feature importance")
    importances = model.feature_importances_
    indices = importances.argsort()[-top_n:][::-1]
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(indices)), importances[indices], align="center")
    plt.yticks(range(len(indices)), [feature_names[i] for i in indices])
    plt.xlabel("Importance")
    plt.title("Top Feature Importances")
    plt.tight_layout()
    if plot_dir:
        _save_plot(plot_dir, f"{model_name}_feature_importance.png")
    plt.show()


def plot_confusion(df, target_col='target', pred_col='prediction', plot_dir=None, model_name="model"):
    logger.info("Plotting confusion matrix")
    cm = confusion_matrix(df[target_col], df[pred_col])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.tight_layout()
    if plot_dir:
        _save_plot(plot_dir, f"{model_name}_confusion_matrix.png")
    plt.show()


def plot_rolling_sharpe(df, window=100, plot_dir=None, model_name="model"):
    logger.info("Plotting rolling Sharpe ratio")
    returns = df['strategy_return']
    rolling_sharpe = returns.rolling(window).mean() / returns.rolling(window).std()
    plt.figure(figsize=(12, 4))
    plt.plot(rolling_sharpe, label=f"Rolling Sharpe ({window})")
    plt.axhline(0, color='gray', linestyle='--')
    plt.title("Rolling Sharpe Ratio")
    plt.xlabel("Time")
    plt.ylabel("Sharpe")
    plt.grid(True)
    plt.tight_layout()
    if plot_dir:
        _save_plot(plot_dir, f"{model_name}_rolling_sharpe.png")
    plt.show()
