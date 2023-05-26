import abc
import functools
import typing
from openff.toolkit import Molecule

import pydantic
from pydantic.main import ModelMetaclass
from openff.nagl._base.metaregistry import create_registry_metaclass
from openff.nagl.nn._metrics import MetricMeta, BaseMetric
from openff.nagl._base.base import ImmutableModel
from openff.nagl.nn._pooling import PoolingLayer
from openff.nagl.nn._containers import ReadoutModule


if typing.TYPE_CHECKING:
    import torch
    from openff.nagl.molecule._dgl import DGLMoleculeOrBatch
    from openff.toolkit import Molecule


class _TargetMeta(ModelMetaclass, abc.ABCMeta, create_registry_metaclass("name")):
    pass


class _BaseTarget(ImmutableModel, abc.ABC, metaclass=_TargetMeta):
    name: typing.ClassVar[typing.Optional[str]] = None

    metric: BaseMetric
    target_label: str
    denominator: float
    weight: float

    @pydantic.validator("metric", pre=True)
    def _validate_metric(cls, v):
        return MetricMeta.get_instance(v)

    @abc.abstractmethod
    def required_columns(self) -> typing.List[str]:
        """
        Target columns used in this target.

        This method is used to determine which columns to extract
        from the parquet datasets.
        """

    def evaluate_loss(
        self,
        molecules: DGLMoleculeOrBatch,
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
        )
        reference = labels[self.target_label]
        loss = self.metric(targets, reference)
        return self.weight * loss / self.denominator

    @abc.abstractmethod
    def evaluate_target(
        self,
        molecules: DGLMoleculeOrBatch,
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
    name: typing.ClassVar[str] = "readout"

    prediction_label: str

    def required_columns(self) -> typing.List[str]:
        return [self.target_label]

    def evaluate_target(
        self,
        molecule: DGLMoleculeOrBatch,
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ) -> "torch.Tensor":
        return predictions[self.target_label]


class HeavyAtomReadoutTarget(_BaseTarget):
    """
    A target that is evaluated on the heavy atoms of the readout of a molecule,
    """
    name: typing.ClassVar[str] = "heavy_atom_readout"
    
    prediction_label: str

    def required_columns(self) -> typing.List[str]:
        return [self.target_label]
    
    def evaluate_target(
        self,
        molecule: DGLMoleculeOrBatch,
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ) -> "torch.Tensor":        
        atomic_numbers = molecule.graph.ndata["atomic_number"]
        heavy_atom_mask = atomic_numbers != 1
        return predictions[self.target_label][heavy_atom_mask]
    

class SingleDipoleTarget(_BaseTarget):
    """A target that is evaluated on the dipole of a molecule."""
    name: typing.ClassVar[str] = "single_dipole"

    charge_label: str
    conformation_column: str

    def required_columns(self) -> typing.List[str]:
        return [self.target_label, self.conformation_column]

    def evaluate_target(
        self,
        molecules: DGLMoleculeOrBatch,
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ) -> "torch.Tensor":
        import torch
        from openff.nagl.molecule._base import BatchMixin


        conformations = labels[self.conformation_column].reshape(-1, 3)
        all_confs = torch.split(
            conformations,
            molecules.n_atoms_per_molecule
        )
        charges = predictions[self.charge_label].squeeze()
        all_charges = torch.split(
            charges,
            molecules.n_atoms_per_molecule
        )
        dipoles = []

        for mol_charge, mol_conformation in zip(all_charges, all_confs):
            mol_dipole = torch.matmul(mol_charge, mol_conformation)
            dipoles.append(mol_dipole)

        return torch.stack(dipoles).squeeze()
    

class MultipleDipoleTarget(_BaseTarget):
    """A target that is evaluated on the dipole of a molecule."""
    name: typing.ClassVar[str] = "multi_dipole"

    charge_label: str
    conformation_column: str
    n_conformation_column: str

    def required_columns(self) -> typing.List[str]:
        return [self.target_label, self.conformation_column, self.n_conformation_column]

    def evaluate_target(
        self,
        molecules: DGLMoleculeOrBatch,
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ) -> "torch.Tensor":
        import torch
        from openff.nagl.molecule._base import BatchMixin


        conformations = labels[self.conformation_column].reshape(-1, 3)
        n_conformations = labels[self.n_conformation_column].reshape(-1)
        n_atoms_per_molecule = molecules.n_atoms_per_molecule
        charges = predictions[self.charge_label].squeeze()
        all_charges = torch.split(
            charges,
            molecules.n_atoms_per_molecule
        )
        dipoles = []

        counter = 0
        for i, n_conf in enumerate(n_conformations):
            mol_charge = all_charges[i]
            n_atoms = n_atoms_per_molecule[i]
            for _ in range(n_conf):
                mol_conformation = conformations[counter:counter+n_atoms]
                mol_dipole = torch.matmul(mol_charge, mol_conformation)
                dipoles.append(mol_dipole)

                counter += n_atoms

        return torch.stack(dipoles).squeeze()
    

class ESPTarget(_BaseTarget):
    """A target that is evaluated on the electrostatic potential of a molecule."""
    name: typing.ClassVar[str] = "esp"

    charge_label: str
    inverse_distance_matrix_column: str
    n_esp_column: str

    def required_columns(self) -> typing.List[str]:
        return [
            self.target_label,
            self.inverse_distance_matrix_column,
            self.n_esp_column
        ]
    
    def evaluate_target(
        self,
        molecules: DGLMoleculeOrBatch,
        labels: typing.Dict[str, "torch.Tensor"],
        predictions: typing.Dict[str, "torch.Tensor"],
        readout_modules: typing.Dict[str, ReadoutModule],
    ) -> "torch.Tensor":
        import torch
        
        inverse_distance_matrix = labels[self.inverse_distance_matrix_column].squeeze()
        n_grid_points = labels[self.n_esp_column]
        charges = predictions[self.charge_label].squeeze()
        all_charges = torch.split(
            charges,
            molecules.n_atoms_per_molecule
        )

        esps = []
        counter = 0
        for n_atoms, n_grid, mol_charge in zip(
            molecules.n_atoms_per_molecule,
            n_grid_points,
            all_charges,
        ):
            increment = n_atoms * n_grid
            inv_dist = inverse_distance_matrix[counter:counter+increment]
            inv_dist = inv_dist.reshape(n_grid, n_atoms)
            esp = torch.matmul(inv_dist, mol_charge)
            esps.extend(esp)

        return torch.tensor(esps).squeeze()

