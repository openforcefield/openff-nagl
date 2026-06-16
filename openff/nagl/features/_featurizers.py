from typing import TYPE_CHECKING, Generic, List, Optional, TypeVar

import torch

from openff.nagl.toolkits.openff import ensure_toolkit_registry

from ._base import Feature
from .atoms import AtomFeature
from .bonds import BondFeature

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule

    from openff.nagl.toolkits.registry import NAGLToolkitRegistry


T = TypeVar("T", bound=Feature)


class Featurizer(Generic[T]):
    features: List[T]

    def __init__(self, features: List[T]):
        self.features = []
        for feature in features:
            if isinstance(feature, type):
                feature = feature()
            self.features.append(feature)

    def featurize(self, molecule: "Molecule", toolkit_registry: Optional["NAGLToolkitRegistry"] = None) -> torch.Tensor:
        toolkit_registry = ensure_toolkit_registry(toolkit_registry)
        encoded = [feature.encode(molecule, toolkit_registry=toolkit_registry) for feature in self.features]
        features = torch.hstack(encoded)
        return features

    __call__ = featurize


class AtomFeaturizer(Featurizer[AtomFeature]):
    pass


class BondFeaturizer(Featurizer[BondFeature]):
    pass
