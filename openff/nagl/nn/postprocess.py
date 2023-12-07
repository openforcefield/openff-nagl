"""
Postprocessing functions to convert a graph representation to a predicted property
"""

import abc
from typing import ClassVar, Dict, Type, Union

import torch

from openff.nagl._base.metaregistry import create_registry_metaclass
from openff.nagl.molecule._dgl import DGLMolecule, DGLMoleculeBatch


class _PostprocessLayerMeta(abc.ABCMeta, create_registry_metaclass()):
    """Metaclass for registering post-processing layers for string lookup."""
    registry: ClassVar[Dict[str, Type]] = {}

    def __init__(cls, name, bases, namespace, **kwargs):
        super().__init__(name, bases, namespace, **kwargs)
        if hasattr(cls, "name") and cls.name:
            cls.registry[cls.name.lower()] = cls

    # @classmethod
    # def get_layer_class(cls, class_name: str):
    #     if isinstance(class_name, cls):
    #         return class_name
    #     if isinstance(type(class_name), cls):
    #         return type(class_name)
    #     try:
    #         return cls.registry[class_name.lower()]
    #     except KeyError:
    #         raise ValueError(
    #             f"Unknown PostprocessLayer type: {class_name}. "
    #             f"Supported types: {list(cls.registry.keys())}"
    #         )


class PostprocessLayer(torch.nn.Module, abc.ABC, metaclass=_PostprocessLayerMeta):
    """A layer to apply to the final readout of a neural network."""

    name: ClassVar[str] = ""
    n_features: ClassVar[int] = 0

    @abc.abstractmethod
    def forward(
        self, molecule: Union[DGLMolecule, DGLMoleculeBatch], inputs: torch.Tensor
    ) -> torch.Tensor:
        """Returns the post-processed input vector."""


class NormalizePartialCharges(PostprocessLayer):
    """
    Maps a set of atomic electronegativity and hardness parameters to partial charges.

    References
    ----------

    1. Gilson, Michael K.; Gilson, Hillary S.R.; Potter, Michael J. "Fast
    assignment of accurate partial atomic charges: an electronegativity
    equalization method that accounts for alternate resonance forms." `Journal
    of Chemical Information and Computer Sciences
    <https://doi.org/10.1021/ci034148o>`_ 43.6 (2003): 1982-1997.

    2. Wang, Yuanqing; Fass, Josh; Stern, Chaya D.; Luo, Kun; Chodera, John
    D. "Graph Nets for Partial Charge Prediction." `arXiv:1909.07903
    [physics.comp-ph] <https://doi.org/10.48550/arXiv.1909.07903>`_
    """

    name: ClassVar[str] = "normalize_partial_charges"
    n_features: ClassVar[int] = 1

    @staticmethod
    def _normalize_partial_charges(
        charges: torch.Tensor,
        total_charge: float,
    ) -> torch.Tensor:
        """
        Ensure molecules have integer charge
        """
        difference = total_charge - charges.sum()
        normalized = charges + (difference / charges.shape[0])
        return normalized.reshape(-1, 1)

    def forward(
        self,
        molecule: Union[DGLMolecule, DGLMoleculeBatch],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        raw_charges = inputs[:, 0]
        formal_charges = molecule.graph.ndata["formal_charge"]

        all_charges = []
        counter = 0
        for n_atoms, n_representations in zip(
            molecule.n_atoms_per_molecule,
            molecule.n_representations_per_molecule,
        ):
            n_atoms = int(n_atoms)
            representation_charges = []
            for i in range(n_representations):
                atom_slice = slice(counter, counter + n_atoms)
                counter += n_atoms

                charges = self._normalize_partial_charges(
                    raw_charges[atom_slice],
                    formal_charges[atom_slice].sum(),
                )
                representation_charges.append(charges)

            mean_charges = torch.stack(representation_charges).mean(dim=0)
            all_charges.append(mean_charges)

        return torch.vstack(all_charges)


class ComputePartialCharges(PostprocessLayer):
    """
    Maps a set of atomic electronegativity and hardness parameters to partial charges.

    References
    ----------

    1. Gilson, Michael K.; Gilson, Hillary S.R.; Potter, Michael J. "Fast
    assignment of accurate partial atomic charges: an electronegativity
    equalization method that accounts for alternate resonance forms." `Journal
    of Chemical Information and Computer Sciences
    <https://doi.org/10.1021/ci034148o>`_ 43.6 (2003): 1982-1997.

    2. Wang, Yuanqing; Fass, Josh; Stern, Chaya D.; Luo, Kun; Chodera, John
    D. "Graph Nets for Partial Charge Prediction." `arXiv:1909.07903
    [physics.comp-ph] <https://doi.org/10.48550/arXiv.1909.07903>`_
    """

    name: ClassVar[str] = "compute_partial_charges"
    n_features: ClassVar[int] = 2

    @staticmethod
    def _calculate_partial_charges(
        electronegativity: torch.Tensor,
        hardness: torch.Tensor,
        total_charge: float,
    ) -> torch.Tensor:
        """
        Equation borrowed from Wang et al's preprint on Espaloma (Eq 15)
        """

        inverse_hardness = 1.0 / hardness
        e_over_s = electronegativity * inverse_hardness
        numerator = e_over_s.sum() + total_charge
        denominator = inverse_hardness.sum()
        fraction = inverse_hardness * (numerator / denominator)

        charges = (-e_over_s + fraction).reshape(-1, 1)
        return charges

    def forward(
        self,
        molecule: Union[DGLMolecule, DGLMoleculeBatch],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        electronegativity = inputs[:, 0]
        hardness = inputs[:, 1]
        formal_charges = molecule.graph.ndata["formal_charge"]

        all_charges = []
        counter = 0
        for n_atoms, n_representations in zip(
            molecule.n_atoms_per_molecule,
            molecule.n_representations_per_molecule,
        ):
            n_atoms = int(n_atoms)
            representation_charges = []
            for i in range(n_representations):
                atom_slice = slice(counter, counter + n_atoms)
                counter += n_atoms

                charges = self._calculate_partial_charges(
                    electronegativity[atom_slice],
                    hardness[atom_slice],
                    formal_charges[atom_slice].sum(),
                )
                representation_charges.append(charges)

            mean_charges = torch.stack(representation_charges).mean(dim=0)
            all_charges.append(mean_charges)

        return torch.vstack(all_charges)

class RegularizedComputePartialCharges(PostprocessLayer):
    """
    Maps a set of initial charges, atomic electronegativity,
    and hardness parameters to partial charges.

    This is a modification of the :class:`ComputePartialCharges`.

    References
    ----------

    1. Gilson, Michael K.; Gilson, Hillary S.R.; Potter, Michael J. "Fast
    assignment of accurate partial atomic charges: an electronegativity
    equalization method that accounts for alternate resonance forms." `Journal
    of Chemical Information and Computer Sciences
    <https://doi.org/10.1021/ci034148o>`_ 43.6 (2003): 1982-1997.

    2. Wang, Yuanqing; Fass, Josh; Stern, Chaya D.; Luo, Kun; Chodera, John
    D. "Graph Nets for Partial Charge Prediction." `arXiv:1909.07903
    [physics.comp-ph] <https://doi.org/10.48550/arXiv.1909.07903>`_
    """

    name: ClassVar[str] = "regularized_compute_partial_charges"
    n_features: ClassVar[int] = 3

    @staticmethod
    def _calculate_partial_charges(
        charge_priors: torch.Tensor,
        electronegativity: torch.Tensor,
        hardness: torch.Tensor,
        total_charge: float,
    ) -> torch.Tensor:
        """
        Equation borrowed from Wang et al's preprint on Espaloma (Eq 15)
        """
        total_prior_charge = charge_priors.sum()
        inverse_hardness = 1.0 / hardness
        e_over_s = electronegativity * inverse_hardness

        numerator = total_prior_charge - total_charge - e_over_s.sum()
        denominator = inverse_hardness.sum()
        fraction = inverse_hardness * (numerator / denominator)

        charges = (charge_priors - e_over_s - fraction).reshape(-1, 1)
        return charges

    def forward(
        self,
        molecule: Union[DGLMolecule, DGLMoleculeBatch],
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        charge_priors = inputs[:, 0]
        electronegativity = inputs[:, 1]
        hardness = inputs[:, 2]
        formal_charges = molecule.graph.ndata["formal_charge"]

        all_charges = []
        counter = 0
        for n_atoms, n_representations in zip(
            molecule.n_atoms_per_molecule,
            molecule.n_representations_per_molecule,
        ):
            n_atoms = int(n_atoms)
            representation_charges = []
            for i in range(n_representations):
                atom_slice = slice(counter, counter + n_atoms)
                counter += n_atoms

                charges = self._calculate_partial_charges(
                    charge_priors[atom_slice],
                    electronegativity[atom_slice],
                    hardness[atom_slice],
                    formal_charges[atom_slice].sum(),
                )
                representation_charges.append(charges)

            mean_charges = torch.stack(representation_charges)
            mean_charges = mean_charges.mean(dim=0)
            all_charges.append(mean_charges)



        return torch.vstack(all_charges)
