from typing import TYPE_CHECKING, Generic, List, TypeVar

import torch

from .atoms import AtomFeature
from ._base import Feature
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
        encoded = [feature.encode(molecule) for feature in self.features]
        features = torch.hstack(encoded)
        return features

    __call__ = featurize


class AtomFeaturizer(Featurizer[AtomFeature]):
    pass


class BondFeaturizer(Featurizer[BondFeature]):
    pass
