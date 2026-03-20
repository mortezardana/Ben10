"""Level 2 combiner: XGBoost meta-learner on OOS predictions."""
import numpy as np
import pandas as pd
from shared.interfaces import PipelineOutput
from utils.logger import get_logger

logger = get_logger("stacking")


class StackingCombiner:
    def __init__(self, config=None):
        self.config = config or {
            'n_estimators': 100, 'max_depth': 3, 'learning_rate': 0.1,
            'random_state': 42,
        }
        self.meta_model = None
        self.pipeline_names = None

    def train(self, pipeline_outputs, market_features, actuals):
        """
        Train meta-model on OOS predictions from child pipelines.

        CRITICAL: Only use OOS predictions — never train on IS predictions.
        """
        from xgboost import XGBClassifier

        self.pipeline_names = list(pipeline_outputs.keys())

        # Build feature matrix: pipeline probabilities + market features
        features = pd.DataFrame(index=actuals.index)
        for name, output in pipeline_outputs.items():
            conf = output.confidence.reindex(actuals.index).fillna(0.5)
            signals = output.signals.reindex(actuals.index).fillna(0)
            features[f'{name}_confidence'] = conf
            features[f'{name}_signal'] = signals

        if market_features is not None:
            mf = market_features.reindex(actuals.index)
            for col in mf.select_dtypes(include='number').columns:
                features[f'market_{col}'] = mf[col]

        features = features.fillna(0)
        y = actuals.values

        self.meta_model = XGBClassifier(**self.config)
        self.meta_model.fit(features.values, y)

        acc = float((self.meta_model.predict(features.values) == y).mean())
        logger.info(f"Stacking meta-model trained: accuracy={acc:.4f}")
        return {'accuracy': acc}

    def combine(self, pipeline_outputs, market_features=None):
        """Generate combined signal from meta-model."""
        features = pd.DataFrame()
        idx = None

        for name in self.pipeline_names:
            output = pipeline_outputs.get(name)
            if output is None:
                continue
            if idx is None:
                idx = output.signals.index
            conf = output.confidence.reindex(idx).fillna(0.5)
            signals = output.signals.reindex(idx).fillna(0)
            features[f'{name}_confidence'] = conf
            features[f'{name}_signal'] = signals

        if market_features is not None:
            mf = market_features.reindex(idx)
            for col in mf.select_dtypes(include='number').columns:
                features[f'market_{col}'] = mf[col]

        features = features.fillna(0)
        probs = self.meta_model.predict_proba(features.values)[:, 1]

        signals = pd.Series(np.where(probs > 0.55, 1, np.where(probs < 0.45, -1, 0)), index=idx)
        confidence = pd.Series(np.abs(probs - 0.5) * 2, index=idx)

        return PipelineOutput(
            signals=signals,
            confidence=confidence,
            metadata={'combiner': 'stacking', 'pipeline_names': self.pipeline_names}
        )
