"""Targets for calculating loss when training neural networks"""

import abc
import pathlib
import typing

import torch
from openff.nagl._base.metaregistry import create_registry_metaclass
from openff.nagl.molecule._dgl import DGLMoleculeOrBatch
from openff.nagl.training.metrics import MetricType #MetricMeta, BaseMetric
from openff.nagl._base.base import ImmutableModel
from openff.nagl.nn._pooling import PoolingLayer
from openff.nagl.nn._containers import ReadoutModule

try:
    from pydantic.v1 import Field, validator
    from pydantic.v1.main import ModelMetaclass
except ImportError:
    from pydantic import Field, validator
    from pydantic.main import ModelMetaclass

if typing.TYPE_CHECKING:
    import torch
    from openff.nagl.molecule._dgl import DGLMoleculeOrBatch
    from openff.toolkit import Molecule


__all__ = [
    "ReadoutTarget",
    "HeavyAtomReadoutTarget",
    "SingleDipoleTarget",
    "MultipleDipoleTarget",
    "MultipleESPTarget",
]

class _BaseTarget(ImmutableModel, abc.ABC): #, metaclass=_TargetMeta):
    name: typing.Literal[""]
    metric: MetricType = Field(..., discriminator="name")
    target_label: str = Field(
        description=(
            "The label to use for the target, or reference property. "
        )
    )
    denominator: float = Field(
        default=1.0,
        description=(
            "The denominator to divide the loss by. This is used to "
            "normalize the loss across targets with different magnitudes."
        )
    )
    weight: float = Field(
        default=1.0,
        description=(
            "The weight to multiply the loss by. This is used to "
            "weight the loss across targets in multi-objective training."
        )
    )

    @validator("metric", pre=True)
    def _validate_metric(cls, v):
        if isinstance(v, str):
            v = {"name": v}
        return v
    
    @abc.abstractmethod
    def get_required_columns(self) -> typing.List[str]:
        """
        Target columns used in this target.

        This method is used to determine which columns to extract
        from the parquet datasets.
        """

    def _evaluate_loss(
        self,
        molecules: "DGLMoleculeOrBatch",
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ):
        """
        Evaluate the target loss for a molecule or batch of molecules.
        This accounts for the denominator and weight of the target and
        calls `evaluate_target`.

        Parameters
        ----------
        molecules: Union[DGLMolecule, DGLMoleculeBatch]
            The molecule(s) to evaluate the target for.
        labels: Dict[str, torch.Tensor]
            The labels for the molecule(s). If `molecules` is a batch,
            the values of these will be contiguous arrays, which
            should be split for molecular-based errors
        predictions: Dict[str, torch.Tensor]
            The predictions for the molecule(s). If `molecules` is a batch,
            the values of these will be contiguous arrays, which
            should be split for molecular-based errors

        Returns
        -------
        torch.Tensor
            The loss for the molecule(s).
        """
        targets = self.evaluate_target(
            molecules, labels, predictions,
            readout_modules=readout_modules
        ).float().squeeze()
        reference = labels[self.target_label].float().squeeze()
        loss = self.metric(targets, reference)
        return loss
    
    def evaluate_loss(
        self,
        molecules: "DGLMoleculeOrBatch",
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ):
        """
        Evaluate the target loss for a molecule or batch of molecules.
        This accounts for the denominator and weight of the target and
        calls `evaluate_target`.

        Parameters
        ----------
        molecules: Union[DGLMolecule, DGLMoleculeBatch]
            The molecule(s) to evaluate the target for.
        labels: Dict[str, torch.Tensor]
            The labels for the molecule(s). If `molecules` is a batch,
            the values of these will be contiguous arrays, which
            should be split for molecular-based errors
        predictions: Dict[str, torch.Tensor]
            The predictions for the molecule(s). If `molecules` is a batch,
            the values of these will be contiguous arrays, which
            should be split for molecular-based errors

        Returns
        -------
        torch.Tensor
            The loss for the molecule(s).
        """
        loss = self._evaluate_loss(
            molecules, labels, predictions, readout_modules
        )
        return self.weight * loss / self.denominator

    @abc.abstractmethod
    def evaluate_target(
        self,
        molecules: "DGLMoleculeOrBatch",
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ) -> "torch.Tensor":
        """
        Evaluate the target property for a molecule or batch of molecules.
        This does *not* need to account for the denominator and weight,
        which will be factored in ``evaluate``.

        Parameters
        ----------
        molecules: DGLMolecule or DGLMoleculeBatch
            The molecule(s) to evaluate the target for.
        labels: Dict[str, torch.Tensor]
            The labels for the molecule(s). If `molecules` is a batch,
            the values of these will be contiguous arrays, which
            should be split for molecular-based errors
        predictions: Dict[str, torch.Tensor]
            The predictions for the molecule(s). If `molecules` is a batch,
            the values of these will be contiguous arrays, which
            should be split for molecular-based errors

        Returns
        -------
        torch.Tensor
            The loss for the molecule(s).
        """

    def compute_reference(self, molecule: "Molecule"):
        raise NotImplementedError
    
    def report_artifact(
        self,
        molecules: "DGLMoleculeOrBatch",
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        output_directory: pathlib.Path,
        top_n_entries: int = 100,
        bottom_n_entries: int = 100,
    ) -> pathlib.Path:
        """
        Create a report of artifacts for this target for an MLFlowLogger.

        Parameters
        ----------
        molecules: DGLMolecule or DGLMoleculeBatch
            The molecule(s) to evaluate the target for.
        labels: Dict[str, torch.Tensor]
            The labels for the molecule(s). If `molecules` is a batch,
            the values of these will be contiguous arrays, which
            should be split for molecular-based errors
        predictions: Dict[str, torch.Tensor]
            The predictions for the molecule(s). If `molecules` is a batch,
            the values of these will be contiguous arrays, which
            should be split for molecular-based errors
        output_directory: pathlib.Path
            The directory to output the report to.
        top_n_entries: int, optional
            The number of top-ranked entries to include in the report.
        bottom_n_entries: int, optional
            The number of bottom-ranked entries to include in the report.

        Returns
        -------
        pathlib.Path
            The path to the report.
        """
        raise NotImplementedError
    
class ReadoutTarget(_BaseTarget):
    """A target that is evaluated on the straightforward readout of a molecule."""
    # name: typing.ClassVar[str] = "readout"
    name: typing.Literal["readout"] = "readout"

    prediction_label: str = Field(
        description=(
            "The predicted property to evaluate the target on."
        )
    )

    def get_required_columns(self) -> typing.List[str]:
        return [self.target_label]

    def evaluate_target(
        self,
        molecules: "DGLMoleculeOrBatch",
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ) -> "torch.Tensor":
        return predictions[self.prediction_label]



class HeavyAtomReadoutTarget(_BaseTarget):
    """
    A target that is evaluated on the heavy atoms of the readout of a molecule,
    """
    # name: typing.ClassVar[str] = "heavy_atom_readout"
    name: typing.Literal["heavy_atom_readout"] = "heavy_atom_readout"
    
    prediction_label: str = Field(
        description=(
            "The predicted property to evaluate the target on."
        )
    )

    def get_required_columns(self) -> typing.List[str]:
        return [self.target_label]
    
    def evaluate_target(
        self,
        molecules: "DGLMoleculeOrBatch",
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ) -> "torch.Tensor":        
        atomic_numbers = molecules.graph.ndata["atomic_number"]
        heavy_atom_mask = atomic_numbers != 1
        return predictions[self.prediction_label].squeeze()[heavy_atom_mask]


class GeneralLinearFitTarget(_BaseTarget):
    """A target that is evaluated to solve the general Ax=b equation."""

    name: typing.Literal["general_linear_fit"] = "general_linear_fit"
    prediction_label: str = Field(
        description=(
            "The predicted property to evaluate the target on."
        )
    )
    design_matrix_column: str = Field(
        description=(
            "The column in the labels that contains the design matrix."
        )
    )

    def get_required_columns(self) -> typing.List[str]:
        return [self.target_label, self.design_matrix_column]
    
    def evaluate_target(
        self,
        molecules: "DGLMoleculeOrBatch",
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ) -> "torch.Tensor":
        x_vectors = predictions[self.prediction_label].squeeze().float()
        A_matrices = labels[self.design_matrix_column].float()
        all_n_atoms = tuple(map(int, molecules.n_atoms_per_molecule))

        result_vectors = []
        # split up by molecule
        for n_atoms in all_n_atoms:
            x_vector = x_vectors[:n_atoms]
            A_matrix = A_matrices[:n_atoms * n_atoms].reshape(n_atoms, n_atoms)
            result = torch.matmul(A_matrix, x_vector)
            result_vectors.append(result)

            x_vectors = x_vectors[n_atoms:]
            A_matrices = A_matrices[n_atoms * n_atoms:]

        return torch.cat(result_vectors)


class SingleDipoleTarget(_BaseTarget):
    """A target that is evaluated on the dipole of a molecule."""
    # name: typing.ClassVar[str] = "single_dipole"
    name: typing.Literal["single_dipole"] = "single_dipole"

    charge_label: str
    conformation_column: str

    def get_required_columns(self) -> typing.List[str]:
        return [self.target_label, self.conformation_column]

    def evaluate_target(
        self,
        molecules: "DGLMoleculeOrBatch",
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ) -> "torch.Tensor":
        import torch

        conformations = labels[self.conformation_column].reshape(-1, 3).float()
        all_n_atoms = tuple(map(int, molecules.n_atoms_per_molecule))
        all_confs = torch.split(conformations, all_n_atoms)
        charges = predictions[self.charge_label].squeeze().float()
        all_charges = torch.split(charges, all_n_atoms)
        dipoles = []

        for mol_charge, mol_conformation in zip(all_charges, all_confs):
            mol_dipole = torch.matmul(mol_charge, mol_conformation)
            dipoles.append(mol_dipole)

        return torch.stack(dipoles).squeeze()
    

class MultipleDipoleTarget(_BaseTarget):
    """A target that is evaluated on the dipole of a molecule."""
    name: typing.Literal["multiple_dipoles"] = "multiple_dipoles"

    charge_label: str
    conformation_column: str
    n_conformation_column: str

    def get_required_columns(self) -> typing.List[str]:
        return [self.target_label, self.conformation_column, self.n_conformation_column]
    
    def _prepare_inputs(
        self,
        molecules: "DGLMoleculeOrBatch",
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
    ):
        import torch

        conformations = labels[self.conformation_column].reshape(-1, 3)
        n_conformations = labels[self.n_conformation_column].reshape(-1).int()
        all_n_atoms = tuple(map(int, molecules.n_atoms_per_molecule))
        charges = predictions[self.charge_label].squeeze()
        all_charges = torch.split(charges, all_n_atoms)
        return conformations, n_conformations, all_n_atoms, all_charges
    
    def evaluate_target(
        self,
        molecules: "DGLMoleculeOrBatch",
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ) -> "torch.Tensor":
        import torch

        conformations, n_conformations, all_n_atoms, all_charges = self._prepare_inputs(
            molecules, labels, predictions
        )
        dipoles = []
        for mol_charge, n_atoms, n_conf in zip(all_charges, all_n_atoms, n_conformations):
            conformer_increment = n_atoms * n_conf
            mol_conformations = conformations[:conformer_increment].reshape(n_conf, n_atoms, 3)
            dipoles.append(torch.matmul(mol_charge, mol_conformations).reshape(-1))
            conformations = conformations[conformer_increment:]

        return torch.cat(dipoles)
    
    
class MultipleESPTarget(_BaseTarget):
    """A target that is evaluated on the electrostatic potential of a molecule."""
    name: typing.Literal["multiple_esps"] = "multiple_esps"

    charge_label: str
    inverse_distance_matrix_column: str
    esp_length_column: str
    n_esp_column: str

    def get_required_columns(self) -> typing.List[str]:
        return [
            self.target_label,
            self.inverse_distance_matrix_column,
            self.esp_length_column,
            self.n_esp_column
        ]
    
    def _prepare_inputs(
        self,
        molecules: "DGLMoleculeOrBatch",
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
    ):
        import torch

        inverse_distance_matrix = labels[self.inverse_distance_matrix_column]
        inverse_distance_matrix = inverse_distance_matrix.squeeze()
        n_grid_points = labels[self.esp_length_column].int()
        all_n_esps = labels[self.n_esp_column].int()
        charges = predictions[self.charge_label].squeeze().to(inverse_distance_matrix.dtype)
        all_n_atoms = tuple(map(int, molecules.n_atoms_per_molecule))
        all_charges = torch.split(charges, all_n_atoms)

        return all_n_atoms, all_n_esps, all_charges, n_grid_points, inverse_distance_matrix
        
    
    def evaluate_target(
        self,
        molecules: "DGLMoleculeOrBatch",
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ) -> "torch.Tensor":
        import torch

        all_n_atoms, all_n_esps, all_charges, n_grid_points, inverse_distance_matrix = self._prepare_inputs(
            molecules, labels, predictions
        )       

        esps = []
        esp_counter = 0
        n_esp_counter = 0

        for n_atoms, n_esps, mol_charge in zip(
            all_n_atoms,
            all_n_esps,
            all_charges
        ):
            # mol_charge = all_charges[atom_counter:atom_counter+n_atoms]
            for i in range(n_esps):
                n_grid = n_grid_points[n_esp_counter]
                grid_increment = n_grid * n_atoms
                inv_dist = inverse_distance_matrix[:grid_increment]
                inv_dist = inv_dist.reshape((n_grid, n_atoms))
                esps.append(torch.matmul(inv_dist, mol_charge).reshape(-1))
                n_esp_counter += 1
                inverse_distance_matrix = inverse_distance_matrix[grid_increment:]
                esp_counter += n_grid

        return torch.cat(esps)
        




TargetType = typing.Union[
    MultipleDipoleTarget,
    ReadoutTarget,
    HeavyAtomReadoutTarget,
    SingleDipoleTarget,
    MultipleESPTarget,
    GeneralLinearFitTarget
]