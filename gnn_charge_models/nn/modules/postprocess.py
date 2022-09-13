import abc
from typing import Union

import torch

from gnn_charge_models.dgl import DGLMolecule, DGLMoleculeBatch


class PostprocessLayer(torch.nn.Module, abc.ABC):
    """A layer to apply to the final readout of a neural network."""

    @abc.abstractmethod
    def forward(
        self,
        molecule: Union[DGLMolecule, DGLMoleculeBatch],
        inputs: torch.Tensor
    ) -> torch.Tensor:
        """Returns the post-processed input vector."""


class ComputePartialCharges(PostprocessLayer):
    """A layer which will map an NN readout containing a set of atomic electronegativity
    and hardness parameters to a set of partial charges [1].

    References:
        [1] Gilson, Michael K., Hillary SR Gilson, and Michael J. Potter. "Fast
            assignment of accurate partial atomic charges: an electronegativity
            equalization method that accounts for alternate resonance forms." Journal of
            chemical information and computer sciences 43.6 (2003): 1982-1997.
    """

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
        for n_atoms, n_representations in zip(
            molecule.n_atoms_per_molecule,
            molecule.n_representations_per_molecule,
        ):
            representation_charges = []
            for i in range(n_representations):
                atom_slice = slice(
                    int(i * n_atoms),
                    int((i + 1) * n_atoms),
                )

            # for i in range(n_representations, counter):
            #     atom_slice = slice(i, i + n_atoms)
                charges = self._calculate_partial_charges(
                    electronegativity[atom_slice],
                    hardness[atom_slice],
                    formal_charges[atom_slice].sum(),
                )
                representation_charges.append(charges)

            mean_charges = torch.stack(representation_charges).mean(dim=0)
            all_charges.append(mean_charges)

        return torch.vstack(all_charges)
