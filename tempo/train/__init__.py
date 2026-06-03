"""Training utilities for TEMPO.

Default trainer is v2 (SFTTrainer + DDP). Set SHRIKE_TRAINER=v1 to use
the old Accelerate trainer.
"""

import os

_trainer_version = os.environ.get("SHRIKE_TRAINER", "v2")

if _trainer_version == "v1":
    from .trainer import train
else:
    from .trainer_v2 import train

from .trainer import train as train_v1
from .trainer_v2 import train as train_v2
from .pipeline import run_pipeline, PipelineConfig
from .parquet_dataset import ParquetMapDataset, load_parquet_splits
from .augment import augment_ts

__all__ = ["train", "train_v1", "train_v2",
           "run_pipeline", "PipelineConfig",
           "ParquetMapDataset", "load_parquet_splits", "augment_ts"]
