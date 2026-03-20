"""Parquet-based feature cache."""

import pandas as pd
from pathlib import Path
from utils.logger import get_logger

logger = get_logger("feature_store")


class FeatureStore:
    def __init__(self, cache_dir="data/features/"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    def get_features(self, data, feature_set_name, version="v1", compute_fn=None):
        """Compute features, cache as Parquet. Return cached if exists."""
        cache_path = self.cache_dir / f"{feature_set_name}_{version}.parquet"

        if cache_path.exists():
            logger.info(f"Loading cached features: {cache_path}")
            return pd.read_parquet(cache_path)

        if compute_fn is None:
            raise ValueError(f"No cached features and no compute_fn provided for {feature_set_name}")

        features = compute_fn(data)
        features.to_parquet(cache_path)
        logger.info(f"Cached features to {cache_path}")
        return features

    def invalidate(self, feature_set_name=None):
        """Clear cache for a feature set or all."""
        if feature_set_name:
            for f in self.cache_dir.glob(f"{feature_set_name}_*.parquet"):
                f.unlink()
                logger.info(f"Invalidated cache: {f}")
        else:
            for f in self.cache_dir.glob("*.parquet"):
                f.unlink()
            logger.info("Invalidated all feature caches")

    def list_feature_sets(self):
        """List available cached feature sets."""
        return [f.stem for f in self.cache_dir.glob("*.parquet")]
