import functools
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from .atoms import AtomFeatureMeta
from .bonds import BondFeatureMeta

_FEATURE_METACLASSES = {
    "atoms": AtomFeatureMeta,
    "bonds": BondFeatureMeta,
}

FeatureType = Union[str, AtomFeatureMeta, BondFeatureMeta]


@functools.total_ordering
class FeatureArgs:
    def __eq__(self, other):
        return (
            self.feature_class == other.feature_class
            and self.feature_arguments == other.feature_arguments
        )

    def __lt__(self, other):
        return self.feature_class.__name__ < other.feature_class.__name__ or (
            self.feature_class.__name__ == other.feature_class.__name__
            and self.feature_arguments < other.feature_arguments
        )

    @classmethod
    def from_input(
        cls,
        feature_input: Union[
            FeatureType, Dict[FeatureType, Any], Tuple[FeatureType, Any]
        ],
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
                feature_class, feature_arguments = list(feature_input.items())[0]

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
        feature_type: Literal["atoms", "bonds"] = "atoms",
    ):

        metacls = _FEATURE_METACLASSES[feature_type.lower()]
        self.feature_class = metacls._get_class(feature_class)
        if feature_arguments is None:
            feature_arguments = tuple()
        self.feature_arguments = tuple(feature_arguments)
        self.feature_type = feature_type

    def __call__(self):
        return self.feature_class(*self.feature_arguments)

    def to_dict(self):
        return {
            "feature_class": self.feature_class.feature_name,
            "feature_arguments": self.feature_arguments,
            "feature_type": self.feature_type,
        }
