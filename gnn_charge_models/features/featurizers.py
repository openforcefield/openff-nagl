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

    features: List[T]

    def __init__(self, features: List[T]):
        self.features = []
        for feature in features:
            if isinstance(feature, type):
                feature = feature()
            self.features.append(feature)

    def featurize(self, molecule: "OFFMolecule") -> torch.Tensor:
        return torch.hstack([
            feature.encode(molecule)
            for feature in self.features
        ])


class AtomFeaturizer(Featurizer[AtomFeature]):
    pass


class BondFeaturizer(Featurizer[BondFeature]):
    pass
