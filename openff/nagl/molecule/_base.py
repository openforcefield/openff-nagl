from typing import ClassVar, TYPE_CHECKING, Tuple, Optional

from openff.nagl.molecule._utils import FEATURE

if TYPE_CHECKING:
    import torch
    from openff.nagl.features.atoms import AtomFeature
    from openff.nagl.features.bonds import BondFeature


class NAGLMoleculeBase:
    _graph_feature_name: ClassVar[str] = "h"
    _graph_forward_edge_type: ClassVar[str] = "forward"
    _graph_backward_edge_type: ClassVar[str] = "reverse"

    def __init__(self, graph):
        self.graph = graph

    @property
    def atom_features(self) -> "torch.Tensor":
        return self.graph.ndata[FEATURE].float()
    
    @property
    def bond_features(self) -> Optional["torch.Tensor"]:
        if FEATURE in self.graph.edata:
            n = int(self.n_bonds)
            key = ("atom", self._graph_forward_edge_type, "atom")
            return self.graph.edata[FEATURE][key][:n].float()
        return None

    @property
    def homograph(self):
        return self.to_homogenous()


class MoleculeMixin:
    def __init__(
            self,
            graph,
            n_representations: int = 1,
            mapped_smiles: str = "",
        ):
        self.graph = graph
        self.n_representations = n_representations
        self.mapped_smiles = mapped_smiles

    @property
    def n_atoms(self):
        return self.n_graph_nodes / self.n_representations

    @property
    def n_atoms_per_molecule(self):
        return (self.n_atoms,)

    @property
    def n_bonds(self):
        return self.n_graph_edges / self.n_representations

    @property
    def n_representations_per_molecule(self):
        return (self.n_representations,)

    @classmethod
    def from_smiles(
        cls,
        smiles: str,
        mapped: bool = False,
        atom_features: Tuple["AtomFeature"] = tuple(),
        bond_features: Tuple["BondFeature"] = tuple(),
        enumerate_resonance_forms: bool = False,
        lowest_energy_only: bool = True,
        max_path_length: Optional[int] = None,
        include_all_transfer_pathways: bool = False,
    ):
        from openff.toolkit import Molecule

        func = Molecule.from_smiles
        if mapped:
            func = Molecule.from_mapped_smiles
        molecule = func(smiles)
        return cls.from_openff(
            molecule=molecule,
            atom_features=atom_features,
            bond_features=bond_features,
            enumerate_resonance_forms=enumerate_resonance_forms,
            lowest_energy_only=lowest_energy_only,
            max_path_length=max_path_length,
            include_all_transfer_pathways=include_all_transfer_pathways,
        )
    
    def to_openff(self):
        from openff.toolkit.topology import Molecule
        from openff.nagl.toolkits.openff import capture_toolkit_warnings

        with capture_toolkit_warnings():
            molecule = Molecule.from_mapped_smiles(
                self.mapped_smiles,
                allow_undefined_stereo=True,
            )
        
        return molecule


class BatchMixin:
    def __init__(
        self, graph, n_representations: Tuple[int, ...], n_atoms: Tuple[int, ...]
    ):
        self.graph = graph
        self.n_representations = n_representations
        self.n_atoms = n_atoms

    @property
    def n_atoms_per_molecule(self):
        return self.n_atoms

    @property
    def n_representations_per_molecule(self):
        return self.n_representations
