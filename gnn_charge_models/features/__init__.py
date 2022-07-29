from typing import Union, Literal, Optional, List, Any, Dict, Tuple

from .atoms import AtomFeatureMeta, AtomFeature
from .bonds import BondFeatureMeta, BondFeature

_FEATURE_METACLASSES = {
    "atoms": AtomFeatureMeta,
    "bonds": BondFeatureMeta,
}

FeatureType = Union[str, AtomFeatureMeta, BondFeatureMeta]


class FeatureArgs:

    @classmethod
    def from_input(
        cls,
        feature_input: Union[FeatureType, Dict[FeatureType, Any], Tuple[FeatureType, Any]],
        feature_type: Literal["atoms", "bonds"] = "atoms",
    ) -> "FeatureArgs":
        if isinstance(feature_input, cls):
            return feature_input
            
        feature_class = None
        if isinstance(feature_input, (str, AtomFeatureMeta, BondFeatureMeta)):
            feature_class = feature_input
            feature_arguments = None

        elif isinstance(feature_input, dict):
            if len(feature_input) == 1:
                feature_class, feature_arguments = list(
                    feature_input.items())[0]

        elif isinstance(feature_input, (list, tuple)):
            if len(feature_input) == 2:
                feature_class, feature_arguments = feature_input

        if feature_class is None:
            raise ValueError(
                "Expected feature_input to be a string, a dictionary with "
                f"one key-value pair, or a tuple with two elements, but got {feature_input}"
            )

        return cls(feature_class, feature_arguments, feature_type=feature_type)

    def __init__(
        self,
        feature_class: FeatureType,
        feature_arguments: Optional[List[Any]] = None,
        feature_type: Literal["atoms", "bonds"] = "atoms"
    ):
        metacls = _FEATURE_METACLASSES[feature_type.lower()]
        self.feature_class = metacls.get_feature_class(feature_class)
        if feature_arguments is None:
            feature_arguments = tuple()
        self.feature_arguments = tuple(feature_arguments)

    def __call__(self):
        return self.feature_class(*self.feature_arguments)
