from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import pandas as pd
import numpy as np


@dataclass
class PipelineOutput:
    """Standard output from any trading pipeline."""
    signals: pd.Series          # -1, 0, +1 (short, flat, long)
    confidence: pd.Series       # 0.0 to 1.0 (calibrated probability)
    metadata: dict = field(default_factory=dict)


@dataclass
class BacktestResult:
    """Standard backtest result."""
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
