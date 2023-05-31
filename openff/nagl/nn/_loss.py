import abc
import typing
from openff.toolkit import Molecule

import pydantic
from pydantic import Field
from pydantic.main import ModelMetaclass
from openff.nagl._base.metaregistry import create_registry_metaclass
from openff.nagl.nn._metrics import MetricType #MetricMeta, BaseMetric
from openff.nagl._base.base import ImmutableModel
from openff.nagl.nn._pooling import PoolingLayer
from openff.nagl.nn._containers import ReadoutModule


if typing.TYPE_CHECKING:
    import torch
    from openff.nagl.molecule._dgl import DGLMoleculeOrBatch
    from openff.toolkit import Molecule


# class _TargetMeta(ModelMetaclass, abc.ABCMeta, create_registry_metaclass("name")):
#     pass


class _BaseTarget(ImmutableModel, abc.ABC): #, metaclass=_TargetMeta):
    name: typing.Literal[""]
    metric: MetricType = Field(..., discriminator="name")
    target_label: str
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

    @pydantic.validator("metric", pre=True)
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
        targets = self.evaluate_target(
            molecules, labels, predictions,
            readout_modules=readout_modules
        ).float()
        reference = labels[self.target_label].float()
        loss = self.metric(targets, reference)
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
    
class ReadoutTarget(_BaseTarget):
    """A target that is evaluated on the straightforward readout of a molecule."""
    # name: typing.ClassVar[str] = "readout"
    name: typing.Literal["readout"] = "readout"

    prediction_label: str

    def get_required_columns(self) -> typing.List[str]:
        return [self.target_label]

    def evaluate_target(
        self,
        molecule: "DGLMoleculeOrBatch",
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ) -> "torch.Tensor":
        return predictions[self.prediction_label].squeeze()


class HeavyAtomReadoutTarget(_BaseTarget):
    """
    A target that is evaluated on the heavy atoms of the readout of a molecule,
    """
    # name: typing.ClassVar[str] = "heavy_atom_readout"
    name: typing.Literal["heavy_atom_readout"] = "heavy_atom_readout"
    
    prediction_label: str

    def get_required_columns(self) -> typing.List[str]:
        return [self.target_label]
    
    def evaluate_target(
        self,
        molecule: "DGLMoleculeOrBatch",
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ) -> "torch.Tensor":        
        atomic_numbers = molecule.graph.ndata["atomic_number"]
        heavy_atom_mask = atomic_numbers != 1
        return predictions[self.prediction_label].squeeze()[heavy_atom_mask]
    

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
    # name: typing.ClassVar[str] = "multi_dipole"
    name: typing.Literal["multi_dipole"] = "multi_dipole"

    charge_label: str
    conformation_column: str
    n_conformation_column: str

    def get_required_columns(self) -> typing.List[str]:
        return [self.target_label, self.conformation_column, self.n_conformation_column]

    def evaluate_target(
        self,
        molecules: "DGLMoleculeOrBatch",
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ) -> "torch.Tensor":
        import torch

        conformations = labels[self.conformation_column].reshape(-1, 3).float()
        n_conformations = labels[self.n_conformation_column].reshape(-1).int()
        all_n_atoms = tuple(map(int, molecules.n_atoms_per_molecule))
        charges = predictions[self.charge_label].squeeze()
        all_charges = torch.split(charges, all_n_atoms)
        dipoles = []

        counter = 0
        for i, n_conf in enumerate(n_conformations):
            mol_charge = all_charges[i]
            n_atoms = all_n_atoms[i]
            for _ in range(n_conf):
                mol_conformation = conformations[counter:counter+n_atoms]
                mol_dipole = torch.matmul(mol_charge, mol_conformation)
                dipoles.extend(mol_dipole)

                counter += n_atoms

        return torch.stack(dipoles).squeeze()
    

class ESPTarget(_BaseTarget):
    """A target that is evaluated on the electrostatic potential of a molecule."""
    # name: typing.ClassVar[str] = "esp"
    name: typing.Literal["esp"] = "esp"

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
    
    def evaluate_target(
        self,
        molecules: "DGLMoleculeOrBatch",
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ) -> "torch.Tensor":
        import torch
        
        inverse_distance_matrix = labels[self.inverse_distance_matrix_column]
        inverse_distance_matrix = inverse_distance_matrix.squeeze().float()
        n_grid_points = labels[self.esp_length_column].int()
        all_n_esps = labels[self.n_esp_column].int()
        charges = predictions[self.charge_label].squeeze().float()
        all_n_atoms = tuple(map(int, molecules.n_atoms_per_molecule))
        all_charges = torch.split(charges, all_n_atoms)

        esps = []
        n_esp_counter = 0
        n_grid_counter = 0
        for n_atoms, n_esps, mol_charge in zip(
            all_n_atoms,
            all_n_esps,
            all_charges,
        ):
            for i in range(n_esps):
                n_grid = n_grid_points[n_esp_counter] * n_atoms
                grid_start = n_grid_counter
                grid_end = grid_start + n_grid
                inv_dist = inverse_distance_matrix[grid_start:grid_end]
                inv_dist = inv_dist.reshape((-1, n_atoms))
                esp = torch.matmul(inv_dist, mol_charge)
                esps.extend(esp)
                n_esp_counter += 1
                n_grid_counter += n_grid

        return torch.tensor(esps).squeeze()




TargetType = typing.Union[
    MultipleDipoleTarget,
    ReadoutTarget,
    HeavyAtomReadoutTarget,
    SingleDipoleTarget,
    ESPTarget,
]