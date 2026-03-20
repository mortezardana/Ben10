"""Level 1 combiner: Weighted voting based on rolling OOS accuracy."""
import numpy as np
import pandas as pd
from shared.interfaces import PipelineOutput
from utils.logger import get_logger

logger = get_logger("weighted_voting")


class WeightedVotingCombiner:
    def __init__(self, lookback=100, min_weight=0.05):
        self.lookback = lookback
        self.min_weight = min_weight
        self.weights = {}

    def combine(self, pipeline_outputs, recent_actuals=None):
        """
        Combine multiple pipeline outputs into one.
        Weights = rolling OOS accuracy if actuals provided, else equal weights.
        """
        names = list(pipeline_outputs.keys())
        if not names:
            raise ValueError("No pipeline outputs to combine")

        # Compute weights
        if recent_actuals is not None and len(recent_actuals) > 0:
            for name, output in pipeline_outputs.items():
                aligned = output.signals.reindex(recent_actuals.index)
                correct = ((aligned > 0) == (recent_actuals > 0)).tail(self.lookback)
                acc = correct.mean() if len(correct) > 0 else 0.5
                self.weights[name] = max(acc, self.min_weight)
        else:
            self.weights = {name: 1.0 / len(names) for name in names}

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

        # Get common index
        idx = pipeline_outputs[names[0]].signals.index
        for name in names[1:]:
            idx = idx.intersection(pipeline_outputs[name].signals.index)

        # Weighted vote
        weighted_signals = pd.Series(0.0, index=idx)
        weighted_confidence = pd.Series(0.0, index=idx)

        for name, output in pipeline_outputs.items():
            w = self.weights.get(name, 0)
            signals = output.signals.reindex(idx).fillna(0)
            conf = output.confidence.reindex(idx).fillna(0)
            weighted_signals += w * signals
            weighted_confidence += w * conf

        # Convert weighted signal to discrete
        final_signals = pd.Series(0, index=idx)
        final_signals[weighted_signals > 0.3] = 1
        final_signals[weighted_signals < -0.3] = -1

        # Reduce confidence on disagreement
        agreement = weighted_signals.abs()
        final_confidence = weighted_confidence * agreement.clip(0, 1)

        return PipelineOutput(
            signals=final_signals,
            confidence=final_confidence,
            metadata={'weights': self.weights, 'combiner': 'weighted_voting'}
        )

    def get_weights(self):
        return self.weights.copy()

    def compute_signal_correlation(self, pipeline_outputs):
        """Pairwise correlation between pipeline signals."""
        names = list(pipeline_outputs.keys())
        signals_df = pd.DataFrame({name: pipeline_outputs[name].signals for name in names})
        return signals_df.corr()
