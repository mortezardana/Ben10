"""3-state Hidden Markov Model for market regime classification."""
import numpy as np
import pandas as pd
import joblib
from shared.interfaces import TradingPipeline, PipelineOutput
from utils.logger import get_logger

logger = get_logger("hmm_pipeline")


class HMMRegimePipeline(TradingPipeline):
    """
    3-state HMM on returns + volatility.
    States: bull (0), choppy (1), bear (2) — auto-assigned by mean return.
    """

    def __init__(self, config=None):
        self.config = config or {'n_components': 3, 'n_iter': 100, 'random_state': 42}
        self.model = None
        self.state_map = {}  # maps HMM state -> regime label

    @property
    def name(self): return "hmm_regime"

    def _prepare_features(self, data):
        """Extract features for HMM: returns, rolling volatility."""
        close = data['Close'] if 'Close' in data.columns else data.iloc[:, 0]
        returns = close.pct_change().fillna(0)
        vol = returns.rolling(20).std().fillna(returns.std())
        features = np.column_stack([returns.values, vol.values])
        return features, returns

    def train(self, train_data, val_data=None):
        from hmmlearn.hmm import GaussianHMM
        features, returns = self._prepare_features(train_data)

        self.model = GaussianHMM(
            n_components=self.config.get('n_components', 3),
            covariance_type='full',
            n_iter=self.config.get('n_iter', 100),
            random_state=self.config.get('random_state', 42),
        )
        self.model.fit(features)

        # Predict states on training data
        states = self.model.predict(features)

        # Auto-assign regime labels based on mean return per state
        state_returns = {}
        for s in range(self.config.get('n_components', 3)):
            mask = states == s
            if mask.any():
                state_returns[s] = returns.values[mask].mean()
            else:
                state_returns[s] = 0

        sorted_states = sorted(state_returns.keys(), key=lambda s: state_returns[s], reverse=True)
        labels = ['bull', 'choppy', 'bear']
        self.state_map = {sorted_states[i]: labels[min(i, len(labels)-1)] for i in range(len(sorted_states))}

        logger.info(f"HMM trained. State mapping: {self.state_map}")
        logger.info(f"State returns: {state_returns}")

        return {'state_returns': state_returns, 'state_map': self.state_map}

    def predict(self, data) -> PipelineOutput:
        features, _ = self._prepare_features(data)
        states = self.model.predict(features)
        posteriors = self.model.predict_proba(features)

        # Convert states to signals
        signals = pd.Series(0, index=data.index)
        confidence = pd.Series(0.0, index=data.index)

        for i, state in enumerate(states):
            regime = self.state_map.get(state, 'choppy')
            if regime == 'bull':
                signals.iloc[i] = 1
            elif regime == 'bear':
                signals.iloc[i] = -1
            else:
                signals.iloc[i] = 0
            confidence.iloc[i] = float(posteriors[i, state])

        return PipelineOutput(
            signals=signals,
            confidence=confidence,
            metadata={'model': 'hmm_regime', 'state_map': self.state_map}
        )

    def evaluate(self, data):
        features, _ = self._prepare_features(data)
        log_likelihood = self.model.score(features)
        output = self.predict(data)
        # For regime model, report state distribution
        state_dist = output.signals.value_counts().to_dict()
        return {'log_likelihood': float(log_likelihood), 'state_distribution': state_dist}

    def save(self, path):
        joblib.dump({'model': self.model, 'state_map': self.state_map, 'config': self.config}, path)

    def load(self, path):
        d = joblib.load(path)
        self.model = d['model']
        self.state_map = d['state_map']
        self.config = d['config']

    def get_regime_features(self, data):
        """Return regime as feature columns for other models."""
        features, _ = self._prepare_features(data)
        states = self.model.predict(features)
        posteriors = self.model.predict_proba(features)

        result = pd.DataFrame(index=data.index)
        result['regime_state'] = states
        for i in range(posteriors.shape[1]):
            result[f'regime_prob_{i}'] = posteriors[:, i]
        return result
