from openff.nagl._base.base import ImmutableModel
from openff.nagl.config.data import DataConfig
from openff.nagl.config.model import ModelConfig
from openff.nagl.config.optimizer import OptimizerConfig
from openff.nagl.utils._types import FromYamlMixin


class TrainingConfig(ImmutableModel, FromYamlMixin):
    model: ModelConfig
    data: DataConfig
    optimizer: OptimizerConfig
