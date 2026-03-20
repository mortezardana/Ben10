import numpy as np
import pandas as pd
import joblib
from shared.interfaces import TradingPipeline, PipelineOutput
from sklearn.metrics import accuracy_score
from utils.logger import get_logger

logger = get_logger("catboost_pipeline")


class CatBoostPipeline(TradingPipeline):
    def __init__(self, config=None):
        self.config = config or {
            'iterations': 300, 'learning_rate': 0.03, 'depth': 5,
            'random_state': 42, 'verbose': 0, 'allow_writing_files': False,
        }
        self.model = None
        self.feature_cols = None

    @property
    def name(self): return "catboost"

    def train(self, train_data, val_data=None):
        from catboost import CatBoostClassifier
        self.feature_cols = [c for c in train_data.columns
                           if c not in ['target', 'date', 'Date', 'prediction']]
        self.feature_cols = train_data[self.feature_cols].select_dtypes(include='number').columns.tolist()

        X_train = train_data[self.feature_cols].values
        y_train = train_data['target'].values
        self.model = CatBoostClassifier(**self.config)

        eval_set = None
        if val_data is not None:
            eval_set = (val_data[self.feature_cols].values, val_data['target'].values)

        self.model.fit(X_train, y_train, eval_set=eval_set)
        acc = accuracy_score(y_train, self.model.predict(X_train))
        return {'accuracy': acc}

    def predict(self, data) -> PipelineOutput:
        X = data[self.feature_cols].values
        probs = self.model.predict_proba(X)[:, 1]
        signals = pd.Series(np.where(probs > 0.55, 1, np.where(probs < 0.45, -1, 0)), index=data.index)
        confidence = pd.Series(np.abs(probs - 0.5) * 2, index=data.index)
        return PipelineOutput(signals=signals, confidence=confidence, metadata={'model': 'catboost'})

    def evaluate(self, data):
        X = data[self.feature_cols].values
        y = data['target'].values
        return {'accuracy': float(accuracy_score(y, self.model.predict(X)))}

    def save(self, path): joblib.dump({'model': self.model, 'feature_cols': self.feature_cols}, path)
    def load(self, path):
        d = joblib.load(path); self.model = d['model']; self.feature_cols = d['feature_cols']
