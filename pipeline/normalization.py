# pipeline/normalization.py
#
# Leak-free feature normalization.
# The scaler is fit ONLY on training data, then applied to the full dataset
# (train + val + test) to prevent information leakage from future data.

from pathlib import Path

import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

from utils.logger import get_logger

logger = get_logger("normalization")


def normalize_features(df, train_end_idx, exclude_cols=None, scaler=None):
    """
    Normalize numerical features without data leakage.

    The scaler is fit exclusively on the training slice ``df.iloc[:train_end_idx]``
    and then used to transform the entire DataFrame.  When a pre-fitted *scaler*
    is supplied (e.g. loaded from disk for inference), the fitting step is skipped
    and the provided scaler is used directly.

    Parameters
    ----------
    df : pd.DataFrame
        Full dataset (train + val + test rows).
    train_end_idx : int
        Integer index that marks the end of the training split (exclusive).
        Everything in ``df.iloc[:train_end_idx]`` is treated as training data.
    exclude_cols : list[str] | None
        Columns to exclude from normalization (e.g. target, date).
        Defaults to ``['target', 'date']`` when *None*.
    scaler : StandardScaler | None
        An already-fitted scaler to reuse (e.g. for inference).  When provided,
        ``train_end_idx`` is still required by signature but the scaler will
        **not** be re-fitted.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with normalized numeric feature columns.
    scaler : StandardScaler
        The fitted scaler (either newly created or the one passed in).
    """
    if exclude_cols is None:
        exclude_cols = ["target", "date"]

    logger.info(
        "Normalizing features (train_end_idx=%d, total_rows=%d)",
        train_end_idx,
        len(df),
    )

    # Identify numeric feature columns to normalize.
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    numeric_cols = (
        df[feature_cols].select_dtypes(include=["number"]).columns.tolist()
    )

    if not numeric_cols:
        logger.warning("No numeric columns found to normalize.")
        if scaler is None:
            scaler = StandardScaler()
        return df, scaler

    if scaler is None:
        # Fit ONLY on training data to avoid leakage.
        scaler = StandardScaler()
        train_slice = df.iloc[:train_end_idx]
        scaler.fit(train_slice[numeric_cols])
        logger.debug(
            "Scaler fit on training data (%d rows, %d columns)",
            len(train_slice),
            len(numeric_cols),
        )
    else:
        logger.debug("Using pre-fitted scaler (skipping fit step)")

    # Transform the entire dataset with the training-fitted scaler.
    df[numeric_cols] = scaler.transform(df[numeric_cols])

    logger.debug("Normalized columns: %s", numeric_cols)
    logger.info("Feature normalization complete")
    return df, scaler


# ---------------------------------------------------------------------------
# Scaler persistence helpers
# ---------------------------------------------------------------------------

def save_scaler(scaler, path):
    """
    Persist a fitted scaler to disk using joblib.

    Parameters
    ----------
    scaler : StandardScaler
        The fitted scaler to save.
    path : str | Path
        Destination file path (e.g. ``models/scaler.joblib``).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(scaler, path)
    logger.info("Scaler saved to %s", path)


def load_scaler(path):
    """
    Load a previously saved scaler from disk.

    Parameters
    ----------
    path : str | Path
        Path to the joblib file produced by :func:`save_scaler`.

    Returns
    -------
    scaler : StandardScaler
        The loaded scaler, ready for ``transform()`` calls.

    Raises
    ------
    FileNotFoundError
        If *path* does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Scaler file not found: {path}")
    scaler = joblib.load(path)
    logger.info("Scaler loaded from %s", path)
    return scaler
