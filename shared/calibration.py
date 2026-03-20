import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import joblib


class ProbabilityCalibrator:
    """Calibrate raw model probabilities to meaningful confidence scores."""

    def __init__(self, method='isotonic'):
        """method: 'isotonic' or 'platt'"""
        self.method = method
        self.calibrator = None

    def fit(self, y_true, y_prob):
        """Fit calibrator on validation set."""
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(y_prob, y_true)
        elif self.method == 'platt':
            self.calibrator = LogisticRegression()
            self.calibrator.fit(y_prob.reshape(-1, 1), y_true)

    def transform(self, y_prob):
        """Apply calibration to raw probabilities."""
        if self.calibrator is None:
            return y_prob
        if self.method == 'platt':
            return self.calibrator.predict_proba(y_prob.reshape(-1, 1))[:, 1]
        return self.calibrator.transform(y_prob)

    def save(self, path):
        joblib.dump(self, path)

    @staticmethod
    def load(path):
        return joblib.load(path)


def calibrate_probabilities(y_true, y_prob, method='isotonic'):
    """Convenience function: fit and return a calibrator."""
    cal = ProbabilityCalibrator(method=method)
    cal.fit(y_true, y_prob)
    return cal


def apply_calibration(raw_probs, calibrator):
    """Apply fitted calibrator to raw probability outputs."""
    if calibrator is None:
        return raw_probs
    return calibrator.transform(raw_probs)
