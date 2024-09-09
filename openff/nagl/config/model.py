"""Config classes for defining a GNNModel"""


import pathlib
import typing

from openff.nagl._base.base import ImmutableModel
from openff.nagl.nn.gcn._base import _GCNStackMeta
from openff.nagl.nn.activation import ActivationFunction
from openff.nagl.features.atoms import DiscriminatedAtomFeatureType
from openff.nagl.features.bonds import DiscriminatedBondFeatureType
from openff.nagl.utils._types import FromYamlMixin

AggregatorType = typing.Literal["mean", "gcn", "pool", "lstm", "sum"]
PostprocessType = typing.Literal["readout", "compute_partial_charges", "regularized_compute_partial_charges"]

try:
    from pydantic.v1 import Field, validator
except ImportError:
    from pydantic import Field, validator

class BaseLayer(ImmutableModel):
    """Base class for single layer in the neural network"""
    hidden_feature_size: int = Field(
        description=(
            "The feature sizes to use for each hidden layer. "
            "Each hidden layer will have the shape "
            "`n_atoms` x `hidden_feature_sizes`."
        )
    )
    activation_function: ActivationFunction = Field(
        description="The activation function to apply for each layer"
    )
    dropout: float = Field(
        default=0.0,
        description="The dropout to apply after each layer"
    )

    @validator("activation_function", pre=True)
    def _validate_activation_function(cls, v):
        return ActivationFunction._get_class(v)


class ConvolutionLayer(BaseLayer):
    """Configuration for a single convolution layer"""
    aggregator_type: AggregatorType = Field(
        default=None,
        description="The aggregator function to apply after each convolution"
    )


class ForwardLayer(BaseLayer):
    """Configuration for a single feedforward layer"""


class ConvolutionModule(ImmutableModel):
    architecture:typing.Literal["SAGEConv", "GINConv"] = Field(
        description="GCN architecture to use"
    )
    layers: typing.List[ConvolutionLayer] = Field(
        description="Configuration for each layer"
    )


class ReadoutModule(ImmutableModel):
    pooling: typing.Literal["atoms", "bonds"]
    layers: typing.List[ForwardLayer] = Field(
        description="Configuration for each layer"
    )
    postprocess: typing.Optional[PostprocessType] = Field(
        description="Optional post-processing layer for prediction"
    )


class ModelConfig(ImmutableModel, FromYamlMixin):
    """
    The configuration class for a GNNModel
    """
    version: typing.Literal["0.1"]
    atom_features: typing.List[DiscriminatedAtomFeatureType] = Field(
        description="Atom features to use"
    )
    bond_features: typing.List[DiscriminatedBondFeatureType] = Field(
        description=(
            "Bond features to use. "
            "Not all architectures support bond features"
        ),
        default_factory=list
    )
    convolution: ConvolutionModule = Field(
        description="Convolution config to pass molecular graph through"
    )
    readouts: typing.Dict[str, ReadoutModule] = Field(
        description="Readout configs to map convolution representation to output"
    )
    
    def to_simple_dict(self):
        """
        Create a simple dictionary representation of the model config

        This simplifies the representation of atom and bond features
        """
        dct = self.dict()
        dct["atom_features"] = tuple(
            [
                {f.feature_name: f.dict(exclude={"feature_name"})}
                for f in self.atom_features
            ]
        )

        dct["bond_features"] = tuple(
            [
                {f.feature_name: f.dict(exclude={"feature_name"})}
                for f in self.bond_features
            ]
        )
        new_dict = dict(dct)
        for k, v in dct.items():
            if isinstance(v, pathlib.Path):
                v = str(v.resolve())
            new_dict[k] = v
        return new_dict
    
    @property
    def n_atom_features(self) -> int:
        """The number of features used to represent an atom"""
        lengths = [len(feature) for feature in self.atom_features]
        n_features = sum(lengths)
        return n_features
