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

from openff.nagl._base.metaregistry import create_registry_metaclass

from .._base.base import ImmutableModel

if TYPE_CHECKING:
    import torch
    from openff.toolkit.topology import Molecule as OFFMolecule


class FeatureMeta(ModelMetaclass, create_registry_metaclass("feature_name")):
    registry: ClassVar[Dict[str, Type]] = {}

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        try:
            key = namespace.get(cls._key_attribute)
        except AttributeError:
            key = None
        else:
            if not key:
                key = name

        if key is not None:
            key = cls._key_transform(key)
            cls.registry[key] = cls
        setattr(cls, cls._key_attribute, key)


class Feature(ImmutableModel, abc.ABC):
    feature_name: ClassVar[Optional[str]] = ""
    _feature_length: ClassVar[int] = 1

    def __init__(self, *args, **kwargs):
        if not kwargs and args:
            if len(self.__fields__) == len(args):
                kwargs = dict(zip(self.__fields__, args))
                args = tuple()
        super().__init__(*args, **kwargs)

    @classmethod
    def _with_args(cls, *args):
        if len(cls.__fields__) != len(args):
            raise ValueError("Wrong number of arguments")

        kwargs = dict(zip(cls.__fields__, args))
        return cls(**kwargs)

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
