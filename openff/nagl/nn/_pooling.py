"""
Pooling layers
==============

A pooling layer is a layer that takes the output of a graph convolutional layer and
produces a single feature vector for each molecule. This is typically done by
aggregating the node features produced by the graph convolutional layer.

In NAGL, pooling layers are implemented as subclasses of `PoolingLayer`.
They are invoked at various stages of the model.


"""

import abc
import functools
import logging
from typing import ClassVar, Dict, Union, TYPE_CHECKING, Iterable

import torch
import torch.nn

from openff.nagl.molecule._dgl import DGLMolecule, DGLMoleculeBatch, DGLMoleculeOrBatch
from openff.nagl.nn._sequential import SequentialLayers

# TODO: make toolkit-agnostic
from rdkit.Chem import rdMolTransforms

if TYPE_CHECKING:
    import dgl
    from openff.toolkit import Molecule


logger = logging.getLogger(__name__)



def _append_internal_coordinate(pooling_layer):
    """A decorator to append internal coordinates to the pooling layer."""

    def wrapper(representations, molecule):
        if pooling_layer._include_internal_coordinates:
            internal_coordinates = pooling_layer._calculate_internal_coordinates(molecule)
            internal_coordinates = internal_coordinates.reshape((-1, 1))
            representations = [
                torch.cat([representation, internal_coordinates], dim=1)
                for representation in representations
            ]
        return representations

    return wrapper


    

class PoolingLayer(torch.nn.Module, abc.ABC):
    """A convenience class for pooling together node feature vectors produced by
    a graph convolutional layer.
    """

    def __init__(
        self,
        layers: SequentialLayers = None,
        pooling_function: callable = torch.add,
    ):
        super().__init__()
        self.layers = layers
        self._pooling_function = pooling_function

    def forward(self, molecule: DGLMoleculeOrBatch, **kwargs) -> torch.Tensor:
        """Returns the pooled feature vector."""
        representations = self._get_final_pooled_representations(molecule, **kwargs)
        # apply layers
        forwarded = [self.layers(h) for h in representations]
        return self._pooling_function(*forwarded)
    
    @abc.abstractmethod
    def get_nvalues_per_molecule(self, molecule: DGLMoleculeOrBatch) -> Iterable[int]:
        """Returns the number of values per molecule."""

    @classmethod
    @abc.abstractmethod
    def get_n_feature_columns(cls, n_input_features: int) -> int:
        raise NotImplementedError

class PoolAtomFeatures(PoolingLayer):
    """A convenience class for pooling the node feature vectors produced by
    a graph convolutional layer.

    This class simply returns the features "h" from the graphs node data.
    """

    name: ClassVar[str] = "atom"

    def forward(self, molecule: DGLMoleculeOrBatch, **kwargs) -> torch.Tensor:
        return molecule.graph.ndata[molecule._graph_feature_name]
    

    def get_nvalues_per_molecule(self, molecule: DGLMoleculeOrBatch) -> Iterable[int]:
        return molecule.n_atoms_per_molecule
    
    @classmethod
    def get_n_feature_columns(cls, n_input_features: int) -> int:
        return n_input_features


class _SymmetricPoolingLayer(PoolingLayer):
    name: ClassVar[str] = ""
    n_atoms: ClassVar[int] = 0

    def __init__(
        self,
        layers: SequentialLayers = None,
        include_internal_coordinates: bool = False,
    ):
        super().__init__(layers)
        self._include_internal_coordinates = include_internal_coordinates

    def _get_final_pooled_representations(self, molecule: DGLMoleculeOrBatch, **kwargs) -> torch.Tensor:
        representations = self._get_pooled_representations(molecule)
        if self._include_internal_coordinates:
            internal_coordinates = self._calculate_internal_coordinates(molecule)
            internal_coordinates = internal_coordinates.reshape((-1, 1))
        else:
            internal_coordinates = torch.zeros(
                (representations[0].shape[0], 1), dtype=torch.float32
            )
        representations = [
            torch.cat([representation, internal_coordinates], dim=1)
            for representation in representations
        ]

        return representations


    def _generate_transposed_pooling_representation(self, molecule: DGLMoleculeOrBatch) -> torch.Tensor:
        indices = self._generate_single_pooling_representation(molecule)
        transposed = []
        if indices:
            n_params = len(indices[0])
            for i in range(n_params):
                transposed.append([index[i] for index in indices])
        t = torch.tensor(transposed, dtype=torch.long)
        molecule._pooling_representations[self.name] = t

    def _generate_single_pooling_representation(self, molecule: DGLMoleculeOrBatch) -> torch.Tensor:
        raise NotImplementedError


    def _get_pooled_representations(self, molecule: DGLMoleculeOrBatch) -> torch.Tensor:
        h_data = molecule.graph.ndata[molecule._graph_feature_name]

        representations = []
        if not self.name in molecule._pooling_representations:
            self._generate_transposed_pooling_representation(molecule)
        forward_indices = molecule._pooling_representations[self.name]
        if forward_indices.shape[1] > 0:
            for row in forward_indices:
                representations.append(h_data[row])
        
        h_forward = torch.cat(representations, dim=1)
        h_reverse = torch.cat(representations[::-1], dim=1)
        return [h_forward, h_reverse]


    def get_nvalues_per_molecule(self, molecule: DGLMoleculeOrBatch) -> Iterable[int]:
        return molecule._n_pooling_representations_per_molecule[self.name]
    
    def _calculate_internal_coordinates_general(self, molecule: DGLMoleculeOrBatch, calculate_function):
        forward_indices = molecule._pooling_representations[self.name]
        arrays = []
        xyz_data = molecule.graph.ndata["xyz"]
        if forward_indices.shape[1] > 0:
            for row in forward_indices:
                arrays.append(xyz_data[row])
        return calculate_function(*arrays)

    @classmethod
    def get_n_feature_columns(cls, n_input_features: int) -> int:
        return (n_input_features * cls.n_atoms) + 1



class PoolBondFeatures(_SymmetricPoolingLayer):
    """A convenience class for pooling the node feature vectors produced by
    a graph convolutional layer into a set of symmetric bond (edge) features.
    """

    name: ClassVar[str] = "bond"
    n_atoms: ClassVar[int] = 2

    # def _generate_single_pooling_representation(self, molecule: "Molecule"):
    #     bond_indices = sorted([
    #         tuple(sorted([bond.atom1_index, bond.atom2_index]))
    #         for bond in molecule.bonds
    #     ])
    #     return bond_indices

    def _generate_single_pooling_representation(self, molecule: DGLMoleculeOrBatch):
        return molecule._get_bonds()

    
    
    def _calculate_internal_coordinates(self, molecule: DGLMoleculeOrBatch):
        from openff.nagl.utils._tensors import calculate_distances
        return self._calculate_internal_coordinates_general(
            molecule, calculate_distances
        )

class PoolAngleFeatures(_SymmetricPoolingLayer):
    """A convenience class for pooling the node feature vectors produced by
    a graph convolutional layer into a set of symmetric angle features.
    """

    name: ClassVar[str] = "angle"
    n_atoms: ClassVar[int] = 3
    
    # def _generate_single_pooling_representation(self, molecule: "Molecule"):
    #     # molecule.angles is just a set of tuples of atoms
    #     angle_indices = []
    #     for angle in molecule.angles:
    #         indices = (
    #             angle[0].molecule_atom_index,
    #             angle[1].molecule_atom_index,
    #             angle[2].molecule_atom_index,
    #         )
    #         if indices[-1] < indices[0]:
    #             indices = indices[::-1]
    #         angle_indices.append(indices)
    #     angle_indices = sorted(angle_indices)
    #     return angle_indices

    def _generate_single_pooling_representation(self, molecule: DGLMoleculeOrBatch):
        return molecule._get_angles()
    
    def _calculate_internal_coordinates(self, molecule: DGLMoleculeOrBatch):
        from openff.nagl.utils._tensors import calculate_angles
        return self._calculate_internal_coordinates_general(
            molecule, calculate_angles
        )
        


class PoolProperTorsionFeatures(_SymmetricPoolingLayer):
    """A convenience class for pooling the node feature vectors produced by
    a graph convolutional layer into a set of symmetric proper torsion features.
    """

    name: ClassVar[str] = "proper_torsion"
    n_atoms: ClassVar[int] = 4

    # def _generate_single_pooling_representation(self, molecule: "Molecule"):
    #     proper_torsion_indices = []
    #     for torsion in molecule.propers:
    #         indices = (
    #             torsion[0].molecule_atom_index,
    #             torsion[1].molecule_atom_index,
    #             torsion[2].molecule_atom_index,
    #             torsion[3].molecule_atom_index,
    #         )
    #         if indices[-1] < indices[0]:
    #             indices = indices[::-1]
    #         proper_torsion_indices.append(indices)
    #     proper_torsion_indices = sorted(proper_torsion_indices)
    #     return proper_torsion_indices

    def _generate_single_pooling_representation(self, molecule: DGLMoleculeOrBatch):
        return molecule._get_dihedrals()
    
    def _calculate_internal_coordinates(self, molecule: DGLMoleculeOrBatch):
        from openff.nagl.utils._tensors import calculate_dihedrals
        return self._calculate_internal_coordinates_general(
            molecule, calculate_dihedrals
        )


class PoolOneFourFeatures(_SymmetricPoolingLayer):
    name: ClassVar[str] = "one_four"

    # def _generate_single_pooling_representation(self, molecule: "Molecule"):
    #     one_four_indices = []
    #     for torsion in molecule.propers:
    #         indices = (
    #             torsion[0].molecule_atom_index,
    #             torsion[3].molecule_atom_index,
    #         )
    #         if indices[-1] < indices[0]:
    #             indices = indices[::-1]
    #         one_four_indices.append(indices)
    #     one_four_indices = sorted(set(one_four_indices))
    #     return one_four_indices
    
    def _calculate_internal_coordinates(self, molecule: DGLMoleculeOrBatch):
        from openff.nagl.utils._tensors import calculate_distances
        return self._calculate_internal_coordinates_general(
            molecule, calculate_distances
        )


def get_pooling_layer_type(layer: Union[str, type]) -> type:
    if isinstance(layer, type) and issubclass(layer, PoolingLayer):
        return layer
    
    LAYER_TYPES = {
        "atom": PoolAtomFeatures,
        "bond": PoolBondFeatures,
        "angle": PoolAngleFeatures,
        "proper_torsion": PoolProperTorsionFeatures,
        "one_four": PoolOneFourFeatures,
    }

    if isinstance(layer, str):
        if layer.endswith("s"):  # remove plural
            layer = layer[:-1]
        if layer in LAYER_TYPES:
            return LAYER_TYPES[layer]
    
    raise NotImplementedError(f"Unsupported pooling layer '{layer}'.")
    

def get_pooling_layer(layer: Union[str, PoolingLayer], **kwargs) -> PoolingLayer:
    if isinstance(layer, PoolingLayer):
        return layer
    elif isinstance(layer, str):
        layer = get_pooling_layer_type(layer)

    if isinstance(layer, type) and issubclass(layer, PoolingLayer):
        return layer(**kwargs)

    raise NotImplementedError(f"Unsupported pooling layer '{layer}'.")