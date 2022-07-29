from typing import NamedTuple, Dict, Tuple, TYPE_CHECKING, List, Union, Optional
from collections import defaultdict

import dgl
import tqdm
import torch
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from gnn_charge_models.dgl.molecule import DGLMolecule
from gnn_charge_models.dgl.batch import DGLMoleculeBatch
from gnn_charge_models.features import AtomFeature, BondFeature
from gnn_charge_models.utils.utils import as_iterable
from .labellers import LabelPrecomputedMolecule, LabelFunctionLike

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule as OFFMolecule
    from gnn_charge_models.storage.record import ChargeMethod, WibergBondOrderMethod
    from gnn_charge_models.storage.store import MoleculeStore
    from gnn_charge_models.storage.record import MoleculeRecord


OpenFFToDGLConverter = Callable[[
    OFFMolecule, List[AtomFeature], List[BondFeature]], DGLMolecule]


class DGLMoleculeDatasetEntry(NamedTuple):
    """A named tuple containing a featurized molecule graph, a tensor of the atom
    features, and a tensor of the molecule label.
    """

    molecule: DGLMolecule
    labels: Dict[str, torch.Tensor]

    @classmethod
    def from_openff_molecule(
        cls,
        openff_molecule: "OFFMolecule",
        atom_features: List[AtomFeature],
        bond_features: List[BondFeature],
        label_function: LabelFunctionLike,
        openff_to_dgl_converter: OpenFFToDGLConverter = DGLMolecule.from_openff_molecule,
    ):
        labels: Dict[str, torch.Tensor] = label_function(openff_molecule)
        dglmol = openff_to_dgl_converter(
            openff_molecule,
            atom_features,
            bond_features,
        )
        return cls(dglmol, labels)


class DGLMoleculeDataset(Dataset):

    def __len__(self):
        return len(self.entries)
    
    def __getitem__(self, index_or_slice):
        return self.entries[index_or_slice]

    def __init__(self, entries: Tuple[DGLMoleculeDatasetEntry, ...] = tuple()):
        self.entries = list(entries)

    @classmethod
    def from_molecule_stores(
        cls,
        molecule_stores: Tuple["MoleculeStore"],
        atom_features: List[AtomFeature],
        bond_features: List[BondFeature],
        partial_charge_method: Optional["ChargeMethod"] = None,
        bond_order_method: Optional["WibergBondOrderMethod"] = None,
        suppress_toolkit_warnings: bool = True,
    ):
        from gnn_charge_models.utils.openff import capture_toolkit_warnings

        if not partial_charge_method and not bond_order_method:
            raise ValueError(
                "Either partial_charge_method or bond_order_method must be "
                "provided."
            )
        molecule_stores = as_iterable(molecule_stores)
        molecule_records: List["MoleculeRecord"] = [
            record
            for store in molecule_stores
            for record in store.retrieve(
                partial_charge_methods=partial_charge_method or [],
                bond_order_methods=bond_order_method or [],
            )
        ]

        entries = []
        labeller = LabelPrecomputedMolecule(
            partial_charge_method=partial_charge_method,
            bond_order_method=bond_order_method
        )
        for record in tqdm.tqdm(molecule_records, desc="featurizing molecules"):
            with capture_toolkit_warnings(run=suppress_toolkit_warnings):
                offmol = record.to_openff(
                    partial_charge_method=partial_charge_method,
                    bond_order_method=bond_order_method,
                )
                entry = DGLMoleculeDatasetEntry.from_openff_molecule(
                    offmol,
                    atom_features,
                    bond_features,
                    labeller,
                )
                entries.append(entry)

        return cls(entries)


class DGLMoleculeDataLoader(DataLoader):

    def __init__(
        self,
        dataset: Union[DGLMoleculeDataset, ConcatDataset],
        batch_size: Optional[int] = 1,
        shuffle: bool = False,
        num_workers: int = 0,
        **kwargs
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate,
            **kwargs
        )

    @staticmethod
    def _collate(graph_entries: List[DGLMoleculeDatasetEntry]):
        if isinstance(graph_entries[0], DGLMolecule):
            graph_entries = [graph_entries]

        molecules, labels = zip(*graph_entries)

        batched_molecules = DGLMoleculeBatch(*molecules)
        batched_labels = defaultdict(list)

        for label_name, label_value in labels.items():
            batched_labels[label_name].append(label_value.reshape(-1, 1))

        batched_labels = {
            k: torch.vstack(v)
            for k, v in batched_labels.items()
        }

        return batched_molecules, batched_labels
