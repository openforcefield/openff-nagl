
import pathlib
import typing

from pydantic import Field, validator

from openff.nagl._base.base import ImmutableModel
from openff.nagl.nn.gcn._base import _GCNStackMeta
from openff.nagl.nn.activation import ActivationFunction
from openff.nagl.features.atoms import AtomFeature
from openff.nagl.features.bonds import BondFeature

AggregatorType = typing.Literal["mean", "gcn", "pool", "lstm", "sum"]
PostprocessType = typing.Literal["readout", "compute_partial_charges"]

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
        default=None,
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

    @validator("postprocess", pre=True)
    def _validate_postprocess(cls, v):
        from openff.nagl.nn.postprocess import _PostprocessLayerMeta
        if v is None:
            return None
        return _PostprocessLayerMeta._get_object(v)


class ModelConfig(ImmutableModel):
    atom_features: typing.List[AtomFeature] = Field(
        description="Atom features to use"
    )
    bond_features: typing.List[BondFeature] = Field(
        description=(
            "Bond features to use. "
            "Not all architectures support bond features"
        )
    )
    convolution: ConvolutionModule = Field(
        description="Convolution config to pass molecular graph through"
    )
    readouts: typing.Dict[str, ReadoutModule] = Field(
        description="Readout configs to map convolution representation to output"
    )

    @validator("atom_features", "bond_features", pre=True)
    def _validate_atom_features(cls, v, field):
        if isinstance(v, dict):
            v = list(v.items())
        all_v = []
        for item in v:
            if isinstance(item, dict):
                all_v.extend(list(item.items()))
            elif isinstance(item, (str, field.type_, type(field.type_))):
                all_v.append((item, {}))
            else:
                all_v.append(item)

        instantiated = []
        for klass, args in all_v:
            if isinstance(klass, (AtomFeature, BondFeature)):
                instantiated.append(klass)
            else:
                klass = type(field.type_)._get_class(klass)
                if not isinstance(args, dict):
                    item = klass._with_args(args)
                else:
                    item = klass(**args)
                instantiated.append(item)
        return instantiated
    
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