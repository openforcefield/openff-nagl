import abc
from typing import Generic, TypeVar, List, TYPE_CHECKING

import torch

from .base import Feature
from .atoms import AtomFeature
from .bonds import BondFeature

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule as OFFMolecule

T = TypeVar("T", bound=Feature)


class Featurizer(Generic[T]):
    def featurize(cls, molecule: "OFFMolecule", features: List[T]) -> torch.Tensor:
        return torch.hstack([
            feature.encode(molecule)
            for feature in features
        ])


class AtomFeaturizer(Featurizer[AtomFeature]):
    pass


class BondFeaturizer(Featurizer[BondFeature]):
    pass
