from collections import defaultdict
from typing import (
    TYPE_CHECKING,
    Callable,
    Collection,
    Dict,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Union,
)

import torch
import tqdm
from openff.toolkit.topology import Molecule as OFFMolecule
from torch.utils.data import ConcatDataset, DataLoader, Dataset

from openff.nagl.dgl.batch import DGLMoleculeBatch
from openff.nagl.dgl.molecule import DGLMolecule
from openff.nagl.features import AtomFeature, BondFeature
from openff.nagl.utils.utils import as_iterable

from .label import EmptyLabeller, LabelFunctionLike, LabelPrecomputedMolecule

if TYPE_CHECKING:

    from openff.nagl.storage.record import (
        ChargeMethod,
        MoleculeRecord,
        WibergBondOrderMethod,
    )
    from openff.nagl.storage.store import MoleculeStore

__all__ = [
    "OpenFFToDGLConverter",
    "DGLMoleculeDatasetEntry",
    "DGLMoleculeDataset",
    "DGLMoleculeDataLoader",
]


OpenFFToDGLConverter = Callable[
    ["OFFMolecule", List[AtomFeature], List[BondFeature]], DGLMolecule
]


class DGLMoleculeDatasetEntry(NamedTuple):
    """A named tuple containing a featurized molecule graph, a tensor of the atom
    features, and a tensor of the molecule label.
    """

    molecule: DGLMolecule
    labels: Dict[str, torch.Tensor]

    @classmethod
    def from_openff(
        cls,
        openff_molecule: "OFFMolecule",
        atom_features: List[AtomFeature],
        bond_features: List[BondFeature],
        label_function: LabelFunctionLike,
        openff_to_dgl_converter: OpenFFToDGLConverter = DGLMolecule.from_openff,
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

    @property
    def n_features(self) -> int:
        """Returns the number of atom features"""
        if not len(self):
            return 0

        return self[0][0].atom_features.shape[1]

    @classmethod
    def from_openff(
        cls,
        molecules: Collection["OFFMolecule"],
        label_function: LabelFunctionLike,
        atom_features: Tuple[AtomFeature] = tuple(),
        bond_features: Tuple[BondFeature] = tuple(),
    ):
        entries = [
            DGLMoleculeDatasetEntry.from_openff(
                mol,
                atom_features,
                bond_features,
                label_function,
            )
            for mol in tqdm.tqdm(molecules, desc="Featurizing molecules")
        ]
        return cls(entries)

    @classmethod
    def from_sdf(
        cls,
        *sdf_files,
        label_function: LabelFunctionLike = EmptyLabeller,
        atom_features: Tuple[AtomFeature] = tuple(),
        bond_features: Tuple[BondFeature] = tuple(),
    ):
        offmols = []
        for file in sdf_files:
            mols = as_iterable(OFFMolecule.from_file(file, file_format="sdf"))
            offmols.extend(mols)

        return cls.from_openff(offmols, label_function, atom_features, bond_features)

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
        from openff.nagl.utils.openff import capture_toolkit_warnings

        if not partial_charge_method and not bond_order_method:
            raise ValueError(
                "Either partial_charge_method or bond_order_method must be " "provided."
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
            bond_order_method=bond_order_method,
        )
        for record in tqdm.tqdm(molecule_records, desc="featurizing molecules"):
            with capture_toolkit_warnings(run=suppress_toolkit_warnings):
                offmol = record.to_openff(
                    partial_charge_method=partial_charge_method,
                    bond_order_method=bond_order_method,
                )
                entry = DGLMoleculeDatasetEntry.from_openff(
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
        **kwargs,
    ):
        super().__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=self._collate,
            **kwargs,
        )

    @staticmethod
    def _collate(graph_entries: List[DGLMoleculeDatasetEntry]):
        if isinstance(graph_entries[0], DGLMolecule):
            graph_entries = [graph_entries]

        molecules, labels = zip(*graph_entries)

        batched_molecules = DGLMoleculeBatch.from_dgl_molecules(molecules)
        batched_labels = defaultdict(list)

        for molecule_labels in labels:
            for label_name, label_value in molecule_labels.items():
                batched_labels[label_name].append(label_value.reshape(-1, 1))

        batched_labels = {k: torch.vstack(v) for k, v in batched_labels.items()}

        return batched_molecules, batched_labels
