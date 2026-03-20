import os
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from shared.interfaces import TradingPipeline, PipelineOutput
from utils.logger import get_logger
from utils.sequence_utils import create_sequences

logger = get_logger("cnn_lstm_pipeline")

SEQUENCE_LENGTH = 32


class CNNLSTMPipeline(TradingPipeline):
    def __init__(self, config=None):
        self.config = config or {}
        self.sequence_length = self.config.get('sequence_length', SEQUENCE_LENGTH)
        self.device = self.config.get('device', None)
        self.model = None
        self.feature_cols = None

    @property
    def name(self) -> str:
        return "cnn_lstm"

    def _get_feature_cols(self, df):
        return df.drop(columns=['Date', 'date', 'target', 'prediction'], errors='ignore') \
            .select_dtypes(include='number').columns.tolist()

    def _make_sequences(self, df):
        X_raw = df[self.feature_cols].values
        y_raw = df['target'].values if 'target' in df.columns else np.zeros(len(df))
        return create_sequences(X_raw, y_raw, sequence_length=self.sequence_length)

    def _raw_predict(self, X_seq):
        """Return raw sigmoid probabilities."""
        return self.model.predict(X_seq, verbose=0).flatten()

    def train(self, train_data, val_data=None):
        from models.cnn_lstm import train_cnn_lstm_model
        self.feature_cols = self._get_feature_cols(train_data)

        self.model, metrics = train_cnn_lstm_model(
            train_data, target_col='target', device=self.device
        )
        logger.info(f"CNN-LSTM train complete: {metrics}")
        return metrics

    def predict(self, data) -> PipelineOutput:
        X_seq, _ = self._make_sequences(data)
        probs = self._raw_predict(X_seq)

        aligned_index = data.index[self.sequence_length:]

        signals = pd.Series(
            np.where(probs > 0.55, 1, np.where(probs < 0.45, -1, 0)),
            index=aligned_index,
        )
        confidence = pd.Series(np.abs(probs - 0.5) * 2, index=aligned_index)

        return PipelineOutput(
            signals=signals,
            confidence=confidence,
            metadata={'model': 'cnn_lstm', 'sequence_length': self.sequence_length},
        )

    def evaluate(self, data):
        X_seq, y_seq = self._make_sequences(data)
        probs = self._raw_predict(X_seq)
        preds = (probs > 0.5).astype(int)
        return {'accuracy': float(accuracy_score(y_seq, preds))}

    def save(self, path):
        import joblib
        self.model.save(path)
        meta_path = path + '.meta'
        joblib.dump({
            'feature_cols': self.feature_cols,
            'sequence_length': self.sequence_length,
        }, meta_path)

    def load(self, path):
        import joblib
        from tensorflow.keras.models import load_model
        self.model = load_model(path)
        meta_path = path + '.meta'
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            self.feature_cols = meta['feature_cols']
            self.sequence_length = meta.get('sequence_length', SEQUENCE_LENGTH)
