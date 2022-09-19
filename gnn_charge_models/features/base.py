import abc
from typing import (
    TYPE_CHECKING,
    Any,
    ClassVar,
    Dict,
    List,
    Optional,
    Type,
    Union,
)

from pydantic import validator
from pydantic.main import ModelMetaclass

from ..base.base import ImmutableModel

if TYPE_CHECKING:
    import torch
    from openff.toolkit.topology import Molecule as OFFMolecule


class FeatureMeta(ModelMetaclass):
    registry: ClassVar[Dict[str, Type]] = {}

    def __init__(self, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        _key = self.feature_name if hasattr(self, "feature_name") else None
        if _key == "":
            _key = name

        self.registry[_key] = self
        self.feature_name = _key

    def get_feature_class(self, feature_name_or_class: Union[str, "FeatureMeta"]):
        if isinstance(feature_name_or_class, self):
            return feature_name_or_class

        try:
            return self.registry[feature_name_or_class]
        except KeyError:
            raise KeyError(
                f"Unknown feature type: {feature_name_or_class}. "
                f"Supported types: {list(self.registry.keys())}"
            )


class Feature(ImmutableModel, abc.ABC):
    feature_name: ClassVar[Optional[str]] = ""
    _feature_length: ClassVar[int] = 1

    def encode(self, molecule: "OFFMolecule") -> "torch.Tensor":
        """
        Encode the molecule feature into a tensor.
        """
        return self._encode(molecule).reshape(self.tensor_shape)

    @abc.abstractmethod
    def _encode(self, molecule: "OFFMolecule") -> "torch.Tensor":
        """
        Encode the molecule feature into a tensor.
        """

    @property
    def tensor_shape(self):
        """
        Return the shape of the feature tensor.
        """
        return (-1, len(self))

    def __call__(self, molecule) -> "torch.Tensor":
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
