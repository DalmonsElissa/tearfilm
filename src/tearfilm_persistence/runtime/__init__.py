"""Training and scoring infrastructure."""

from tearfilm_persistence.runtime.checkpoint import (
    FORMAT_VERSION,
    CheckpointState,
    load_checkpoint,
    save_checkpoint,
    sha256,
)
from tearfilm_persistence.runtime.config import deep_merge, flatten, load_experiment, read_yaml
from tearfilm_persistence.runtime.factory import build_model, model_config, train_config
from tearfilm_persistence.runtime.loop import (
    EpochRecord,
    TrainConfig,
    TrainingHistory,
    build_optimizer,
    build_scheduler,
    score_loader,
    train_model,
)
from tearfilm_persistence.runtime.seeding import set_seed

__all__ = [
    "FORMAT_VERSION",
    "CheckpointState",
    "EpochRecord",
    "TrainConfig",
    "TrainingHistory",
    "build_model",
    "build_optimizer",
    "build_scheduler",
    "deep_merge",
    "flatten",
    "load_checkpoint",
    "load_experiment",
    "model_config",
    "read_yaml",
    "save_checkpoint",
    "score_loader",
    "set_seed",
    "sha256",
    "train_config",
    "train_model",
]
