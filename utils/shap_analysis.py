import numpy as np
import pandas as pd
from utils.logger import get_logger

logger = get_logger("shap_analysis")


def compute_shap_values(model, X_test, feature_names=None):
    """
    Compute SHAP values for an XGBoost model.

    Parameters:
    - model: trained XGBoost model
    - X_test: numpy array or DataFrame of test features
    - feature_names: list of feature names

    Returns:
    - dict with 'shap_values', 'feature_importance' (sorted), 'expected_value'
    """
    import shap

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)

    # Handle multi-output (binary classification returns list of 2)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use positive class

    if feature_names is None:
        if isinstance(X_test, pd.DataFrame):
            feature_names = X_test.columns.tolist()
        else:
            feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]

    # Mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs_shap, index=feature_names).sort_values(ascending=False)

    logger.info(f"SHAP values computed for {len(feature_names)} features")

    return {
        'shap_values': shap_values,
        'feature_importance': importance,
        'expected_value': explainer.expected_value,
    }


def plot_shap_summary(shap_values, X_test, feature_names, save_path=None):
    """SHAP summary plot (beeswarm)."""
    import shap
    import matplotlib.pyplot as plt

    if isinstance(X_test, np.ndarray):
        X_df = pd.DataFrame(X_test, columns=feature_names)
    else:
        X_df = X_test

    plt.figure()
    shap.summary_plot(shap_values, X_df, show=False)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"SHAP summary plot saved to {save_path}")
    plt.close()


def plot_shap_importance(shap_values, feature_names, top_n=20, save_path=None):
    """Bar chart of mean |SHAP| values."""
    import matplotlib.pyplot as plt

    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=True)
    importance = importance.tail(top_n)

    plt.figure(figsize=(10, max(6, top_n * 0.3)))
    importance.plot(kind='barh')
    plt.xlabel('Mean |SHAP value|')
    plt.title(f'Top {top_n} Feature Importance (SHAP)')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        logger.info(f"SHAP importance plot saved to {save_path}")
    plt.close()


def get_top_features(shap_values, feature_names, n=30):
    """Return top-N features by mean absolute SHAP value."""
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = pd.Series(mean_abs, index=feature_names).sort_values(ascending=False)
    top = importance.head(n).index.tolist()
    logger.info(f"Top {n} SHAP features: {top[:5]}...")
    return top
