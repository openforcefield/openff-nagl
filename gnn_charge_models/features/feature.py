import abc
import copy
from typing import TYPE_CHECKING, List, Any, Tuple, ClassVar, Dict, Type

from pydantic import validator
from pydantic.main import ModelMetaclass

from ..base import ImmutableModel

if TYPE_CHECKING:
    import torch


class FeatureMeta(ModelMetaclass):
    registry: ClassVar[Dict[str, Type]] = {}

    def __init__(self, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        if hasattr(self, "feature_name") and self.feature_name:
            self.registry[self.feature_name] = self


class Feature(ImmutableModel, abc.ABC):
    feature_name: ClassVar[str]
    _feature_length: ClassVar[int] = 1

    def encode(self, molecule) -> torch.Tensor:
        """
        Encode the molecule feature into a tensor.
        """
        return self._encode(molecule).reshape(self.tensor_shape)

    @abc.abstractmethod
    def _encode(self, molecule) -> torch.Tensor:
        """
        Encode the molecule feature into a tensor.
        """

    @property
    def tensor_shape(self):
        """
        Return the shape of the feature tensor.
        """
        return (-1, len(self))

    def __call__(self, molecule) -> torch.Tensor:
        return self.encode(molecule)

    def __len__(self):
        """
        Return the length of the feature.
        """
        return self._feature_length


class CategoricalMixin:
    """
    Mixin class for categorical features.
    """

    categories: List[Any]

    @property
    def _default_categories(self):
        return self.__fields__["categories"].default

    def __len__(self):
        return len(self.categories)
