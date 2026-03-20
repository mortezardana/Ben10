from pathlib import Path
from typing import Optional
from pydantic import BaseModel, field_validator
import yaml


class LabelingConfig(BaseModel):
    method: str = "triple_barrier"
    tp_multiplier: float = 2.0
    sl_multiplier: float = 1.0
    max_holding_period: int = 12


class SamplingConfig(BaseModel):
    method: str = "all"  # 'all' or 'cusum'
    threshold: Optional[float] = None


class DataConfig(BaseModel):
    data_path: Path = Path("Data/gold_4h.csv")
    target_type: str = "classification"
    target_horizon: int = 1
    test_size: float = 0.2
    val_size: float = 0.1
    exclude_cols: list[str] = ["target", "date", "Date", "future_returns", "signal"]
    labeling: LabelingConfig = LabelingConfig()
    sampling: SamplingConfig = SamplingConfig()

    @field_validator('test_size', 'val_size')
    @classmethod
    def validate_size(cls, v):
        if not 0 < v < 1:
            raise ValueError("Size must be between 0 and 1")
        return v


class PipelineConfig(BaseModel):
    name: str
    model_type: str
    params: dict = {}
    features: Optional[list[str]] = None


class CombinerConfig(BaseModel):
    method: str = "weighted_voting"
    confidence_threshold: float = 0.6


class AppConfig(BaseModel):
    data: DataConfig = DataConfig()
    pipelines: list[PipelineConfig] = []
    combiner: CombinerConfig = CombinerConfig()
    random_state: int = 42
    environment: str = "dev"


def load_config(path: str = "configs/default.yaml") -> AppConfig:
    """Load config from YAML, validate with Pydantic."""
    with open(path) as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig(**raw)
