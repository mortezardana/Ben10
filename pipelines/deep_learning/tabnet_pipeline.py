import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from shared.interfaces import TradingPipeline, PipelineOutput
from utils.logger import get_logger

logger = get_logger("tabnet_pipeline")


class TabNetPipeline(TradingPipeline):
    def __init__(self, config=None):
        self.config = config or {}
        self.device_name = self.config.get('device_name', 'auto')
        self.model = None
        self.feature_cols = None

    @property
    def name(self) -> str:
        return "tabnet"

    def _get_feature_cols(self, df):
        return df.drop(columns=['Date', 'date', 'target', 'prediction'], errors='ignore') \
            .select_dtypes(include='number').columns.tolist()

    def train(self, train_data, val_data=None):
        from models.tabnet_model import train_tabnet_model
        self.feature_cols = self._get_feature_cols(train_data)

        self.model, metrics = train_tabnet_model(
            train_data, target_col='target', device_name=self.device_name
        )
        logger.info(f"TabNet train complete: {metrics}")
        return metrics

    def predict(self, data) -> PipelineOutput:
        X = data[self.feature_cols].values
        # TabNet predict_proba returns (n_samples, n_classes)
        probs = self.model.predict_proba(X)[:, 1]

        signals = pd.Series(
            np.where(probs > 0.55, 1, np.where(probs < 0.45, -1, 0)),
            index=data.index,
        )
        confidence = pd.Series(np.abs(probs - 0.5) * 2, index=data.index)

        return PipelineOutput(
            signals=signals,
            confidence=confidence,
            metadata={'model': 'tabnet'},
        )

    def evaluate(self, data):
        X = data[self.feature_cols].values
        y = data['target'].values
        preds = self.model.predict(X)
        return {'accuracy': float(accuracy_score(y, preds))}

    def save(self, path):
        import joblib
        # TabNet has its own save/load; store the model + metadata
        self.model.save_model(path)
        meta_path = path + '.meta'
        joblib.dump({'feature_cols': self.feature_cols}, meta_path)

    def load(self, path):
        import joblib
        from pytorch_tabnet.tab_model import TabNetClassifier
        self.model = TabNetClassifier()
        self.model.load_model(path)
        meta_path = path + '.meta'
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            self.feature_cols = meta['feature_cols']
