from openff.nagl._base.base import ImmutableModel
from openff.nagl.config.data import DataConfig
from openff.nagl.config.model import ModelConfig
from openff.nagl.config.optimizer import OptimizerConfig


class TrainingConfig(ImmutableModel):
    model: ModelConfig
    data: DataConfig
    optimizer: OptimizerConfig