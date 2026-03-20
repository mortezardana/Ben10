import numpy as np
import pandas as pd
import joblib
from shared.interfaces import TradingPipeline, PipelineOutput
from sklearn.metrics import accuracy_score
from utils.logger import get_logger

logger = get_logger("xgboost_pipeline")


class XGBoostPipeline(TradingPipeline):
    def __init__(self, config=None):
        self.config = config or {
            'n_estimators': 300, 'learning_rate': 0.03, 'max_depth': 5,
            'subsample': 0.8, 'colsample_bytree': 0.8, 'eval_metric': 'logloss',
            'random_state': 42, 'use_label_encoder': False,
        }
        self.model = None
        self.feature_cols = None

    @property
    def name(self) -> str:
        return "xgboost"

    def train(self, train_data, val_data=None):
        from xgboost import XGBClassifier
        self.feature_cols = [c for c in train_data.columns
                           if c not in ['target', 'date', 'Date', 'prediction']]
        self.feature_cols = train_data[self.feature_cols].select_dtypes(include='number').columns.tolist()

        X_train = train_data[self.feature_cols].values
        y_train = train_data['target'].values

        self.model = XGBClassifier(**self.config)

        fit_params = {}
        if val_data is not None:
            X_val = val_data[self.feature_cols].values
            y_val = val_data['target'].values
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False

        self.model.fit(X_train, y_train, **fit_params)

        preds = self.model.predict(X_train)
        acc = accuracy_score(y_train, preds)
        logger.info(f"XGBoost train accuracy: {acc:.4f}")
        return {'accuracy': acc}

    def predict(self, data) -> PipelineOutput:
        X = data[self.feature_cols].values
        probs = self.model.predict_proba(X)[:, 1]
        signals = pd.Series(np.where(probs > 0.55, 1, np.where(probs < 0.45, -1, 0)), index=data.index)
        confidence = pd.Series(np.abs(probs - 0.5) * 2, index=data.index)  # scale to 0-1
        return PipelineOutput(signals=signals, confidence=confidence, metadata={'model': 'xgboost'})

    def evaluate(self, data):
        X = data[self.feature_cols].values
        y = data['target'].values
        preds = self.model.predict(X)
        return {'accuracy': float(accuracy_score(y, preds))}

    def save(self, path):
        joblib.dump({'model': self.model, 'feature_cols': self.feature_cols}, path)

    def load(self, path):
        data = joblib.load(path)
        self.model = data['model']
        self.feature_cols = data['feature_cols']
