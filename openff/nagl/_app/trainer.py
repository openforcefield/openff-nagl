import os
import pathlib
from typing import Optional, Tuple, List, Any

import pytorch_lightning as pl
import rich
from pydantic import validator
from typing import List, Union
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from rich import pretty
from rich.console import NewLine

from openff.nagl._base import ImmutableModel
from openff.nagl.features.atoms import AtomFeature
from openff.nagl.features.bonds import BondFeature
from openff.nagl.nn._models import GNNModel
from openff.nagl.nn.dataset import DGLMoleculeLightningDataModule
from openff.nagl.storage.record import ChargeMethod, WibergBondOrderMethod
from openff.nagl.utils._types import FromYamlMixin

from openff.nagl.nn._models import GNNModel

from openff.nagl.config.data import DataConfig
from openff.nagl.config.model import ModelConfig

class Trainer(ImmutableModel, FromYamlMixin):
    model: ModelConfig
    data: DataConfig
    