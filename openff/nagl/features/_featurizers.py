from typing import TYPE_CHECKING, Generic, List, TypeVar

import torch

from .atoms import AtomFeature
from ._base import Feature
from .bonds import BondFeature

from openff.nagl.toolkits.openff import validate_toolkit_registry
from openff.nagl.toolkits import NAGLToolkitRegistry

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule


T = TypeVar("T", bound=Feature)


class Featurizer(Generic[T]):
    features: List[T]

    def __init__(self, features: List[T]):
        self.features = []
        for feature in features:
            if isinstance(feature, type):
                feature = feature()
            self.features.append(feature)

    @validate_toolkit_registry
    def featurize(self, molecule: "Molecule", toolkit_registry: NAGLToolkitRegistry | None = None) -> torch.Tensor:
        encoded = [feature.encode(molecule, toolkit_registry=toolkit_registry) for feature in self.features]
        features = torch.hstack(encoded)
        return features

    __call__ = featurize


class AtomFeaturizer(Featurizer[AtomFeature]):
    pass


class BondFeaturizer(Featurizer[BondFeature]):
    pass
