import contextlib
import copy
import functools
from typing import TYPE_CHECKING, Tuple, List, Union, Dict, NamedTuple, Any, Optional

import numpy as np

from openff.units import unit

from openff.utilities import requires_package
from openff.nagl.toolkits import NAGL_TOOLKIT_REGISTRY
from openff.utilities.exceptions import MissingOptionalDependencyError

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule
    from openff.nagl.utils._types import HybridizationType

class _AtomAttributes(NamedTuple):
    atomic_number: int
    formal_charge: int
    is_aromatic: bool
    stereochemistry: Optional[str]


class _BondAttributes(NamedTuple):
    bond_order: int
    is_aromatic: bool
    stereochemistry: Optional[str]


class _MoleculeGraph(NamedTuple):
    atoms: dict[int, dict[str, Any]]
    bonds: dict[Tuple[int, int], dict[str, Any]]

    def copy(self):
        atoms = {k: dict(v) for k, v in self.atoms.items()}
        bonds = {k: dict(v) for k, v in self.bonds.items()}
        return _MoleculeGraph(atoms=atoms, bonds=bonds)




def call_toolkit_function(function_name, toolkit_registry, *args, **kwargs):
    """
    Call a function from a toolkit wrapper or toolkit registry.

    Parameters
    ----------

    function: function
        The function to call.
    toolkit_registry: ToolkitWrapperBase or ToolkitRegistry
        The toolkit wrapper or registry to call the function from.
    *args:
        The positional arguments to pass to the function.
    **kwargs:
        The keyword arguments to pass to the function.
    """
    from openff.nagl.toolkits.registry import NAGLToolkitRegistry
    from openff.nagl.toolkits._base import (
        NAGLToolkitWrapperMeta,
        NAGLToolkitWrapperBase,
    )
    from openff.toolkit.utils.exceptions import InvalidToolkitRegistryError

    if isinstance(toolkit_registry, NAGLToolkitWrapperMeta):
        toolkit_registry = toolkit_registry()

    if isinstance(toolkit_registry, NAGLToolkitWrapperBase):
        toolkit_function = getattr(toolkit_registry, function_name)
        return toolkit_function(*args, **kwargs)
    elif isinstance(toolkit_registry, NAGLToolkitRegistry):
        return toolkit_registry.call(function_name, *args, **kwargs)
    else:
        raise InvalidToolkitRegistryError(
            "toolkit_registry must be instance of OpenFF NAGL "
            "ToolkitWrapperBase or ToolkitRegistry. "
            f"Given: {toolkit_registry}"
        )


def toolkit_registry_function(function):
    @functools.wraps(function)
    def wrapper(*args, toolkit_registry=NAGL_TOOLKIT_REGISTRY, **kwargs):
        return call_toolkit_function(
            function.__name__, toolkit_registry, *args, **kwargs
        )

    return wrapper


@contextlib.contextmanager
@requires_package("openeye.oechem")
def capture_oechem_warnings():  # pragma: no cover
    from openeye import oechem

    output_stream = oechem.oeosstream()
    oechem.OEThrow.SetOutputStream(output_stream)
    oechem.OEThrow.Clear()

    yield

    oechem.OEThrow.SetOutputStream(oechem.oeerr)


@contextlib.contextmanager
def capture_toolkit_warnings(run: bool = True):  # pragma: no cover
    """A convenience method to capture and discard any warning produced by external
    cheminformatics toolkits excluding the OpenFF toolkit. This should be used with
    extreme caution and is only really intended for use when processing tens of
    thousands of molecules at once."""

    import logging
    import warnings

    if not run:
        yield
        return

    toolkit_logger = logging.getLogger("openff.toolkit")
    openff_logger_level = toolkit_logger.getEffectiveLevel()
    toolkit_logger.setLevel(logging.ERROR)

    try:
        with capture_oechem_warnings():
            yield
    except MissingOptionalDependencyError:
        yield

    toolkit_logger.setLevel(openff_logger_level)


@contextlib.contextmanager
@toolkit_registry_function
def stream_molecules_to_file(
    file: str,
    toolkit_registry=NAGL_TOOLKIT_REGISTRY,
):
    """
    Stream molecules to an SDF file using a context manager.

    Parameters
    ----------
    file: str
        The path to the SDF file to stream molecules to.
    toolkit_registry:
        The toolkit registry to use to write the molecules.

    Examples
    --------

    >>> from openff.toolkit.topology import Molecule
    >>> from openff.toolkit.utils.openff import stream_molecules_to_file
    >>> molecule1 = Molecule.from_smiles("CCO")
    >>> molecule2 = Molecule.from_smiles("CCC")
    >>> with stream_molecules_to_file("molecules.sdf") as writer:
    ...     writer(molecule1)
    ...     writer(molecule2)

    """
    pass


@toolkit_registry_function
def get_molecule_hybridizations(
    molecule: "Molecule",
    toolkit_registry=NAGL_TOOLKIT_REGISTRY,
) -> List["HybridizationType"]:
    """
    Get the hybridization of each atom in a molecule.

    Parameters
    ----------
    molecule: openff.toolkit.topology.Molecule
        The molecule to get the hybridizations of.
    toolkit_registry:
        The toolkit registry to use

    Returns
    -------
    hybridizations: List[HybridizationType]
        The hybridization of each atom in the molecule.
    """
    pass


@toolkit_registry_function
def get_atoms_are_in_ring_size(
    molecule: "Molecule",
    ring_size: int,
    toolkit_registry=NAGL_TOOLKIT_REGISTRY,
) -> List[bool]:
    """
    Determine whether each atom in a molecule is in a ring of a given size.

    Parameters
    ----------
    molecule: openff.toolkit.topology.Molecule
        The molecule to compute ring perception for
    ring_size: int
        The size of the ring to check for.
    toolkit_registry:
        The toolkit registry to use

    Returns
    -------
    in_ring_size: List[bool]

    """
    pass


@toolkit_registry_function
def get_bonds_are_in_ring_size(
    molecule: "Molecule",
    ring_size: int,
    toolkit_registry=NAGL_TOOLKIT_REGISTRY,
) -> List[bool]:
    """
    Determine whether each bond in a molecule is in a ring of a given size.

    Parameters
    ----------
    molecule: openff.toolkit.topology.Molecule
        The molecule to compute ring perception for
    ring_size: int
        The size of the ring to check for.
    toolkit_registry:
        The toolkit registry to use

    Returns
    -------
    in_ring_size: List[bool]
        Bonds are in the same order as the molecule's ``bonds`` attribute.
    """
    pass


@toolkit_registry_function
def get_best_rmsd(
    molecule: "Molecule",
    reference_conformer: Union[np.ndarray, unit.Quantity],
    target_conformer: Union[np.ndarray, unit.Quantity],
    toolkit_registry=NAGL_TOOLKIT_REGISTRY,
) -> unit.Quantity:
    """
    Compute the lowest all-atom RMSD between a reference and target conformer,
    allowing for symmetry-equivalent atoms to be permuted.

    Parameters
    ----------
    molecule: openff.toolkit.topology.Molecule
        The molecule to compute the RMSD for
    reference_conformer: np.ndarray or openff.units.unit.Quantity
        The reference conformer to compare to the target conformer.
        If a numpy array, it is assumed to be in units of angstrom.
    target_conformer: np.ndarray or openff.units.unit.Quantity
        The target conformer to compare to the reference conformer.
        If a numpy array, it is assumed to be in units of angstrom.
    toolkit_registry:
        The toolkit registry to use

    Returns
    -------
    rmsd: unit.Quantity

    Examples
    --------
    >>> from openff.units import unit
    >>> from openff.toolkit.topology import Molecule
    >>> from openff.toolkit.utils.openff import get_best_rmsd
    >>> molecule = Molecule.from_smiles("CCCCO")
    >>> molecule.generate_conformers(n_conformers=2)
    >>> rmsd = get_best_rmsd(molecule, molecule.conformers[0], molecule.conformers[1])
    >>> print(f"RMSD in angstrom: {rmsd.m_as(unit.angstrom)}")

    """


@toolkit_registry_function
def calculate_circular_fingerprint_similarity(
    molecule: "Molecule",
    reference_molecule: "Molecule",
    radius: int = 3,
    nbits: int = 2048,
    toolkit_registry=NAGL_TOOLKIT_REGISTRY,
) -> float:
    """
    Compute the similarity between two molecules using a fingerprinting method.
    Uses a Morgan fingerprint with RDKit and a Circular fingerprint with OpenEye.

    Parameters
    ----------
    molecule: openff.toolkit.topology.Molecule
        The molecule to compute the fingerprint for.
    reference_molecule: openff.toolkit.topology.Molecule
        The molecule to compute the fingerprint for.
    radius: int, default 3
        The radius of the fingerprint to use.
    nbits: int, default 2048
        The length of the fingerprint to use. Not used in RDKit.
    toolkit_registry:
    The toolkit registry to use

    Returns
    -------
    similarity: float
        The Dice similarity between the two molecules.

    """


def is_conformer_identical(
    molecule: "Molecule",
    reference_conformer: Union[np.ndarray, unit.Quantity],
    target_conformer: Union[np.ndarray, unit.Quantity],
    atol: float = 1.0e-3,
) -> bool:
    """
    Determine if two conformers are identical with an RMSD tolerance,
    allowing for symmetry-equivalent atoms to be permuted.

    Parameters
    ----------
    molecule: openff.toolkit.topology.Molecule
        The molecule to compute the RMSD for
    reference_conformer: np.ndarray or openff.units.unit.Quantity
        The reference conformer to compare to the target conformer.
        If a numpy array, it is assumed to be in units of angstrom.
    target_conformer: np.ndarray or openff.units.unit.Quantity
        The target conformer to compare to the reference conformer.
        If a numpy array, it is assumed to be in units of angstrom.
    atol: float, default=1.0e-3
        The absolute tolerance to use when comparing the RMSD.
        This is given in angstrom.

    Returns
    -------
    is_identical: bool
    """

    rmsd = get_best_rmsd(molecule, reference_conformer, target_conformer)
    return rmsd.m_as(unit.angstrom) < atol


def normalize_molecule(
    molecule: "Molecule",
    max_iter: int = 200,
    inplace: bool = False,
    toolkit_registry=NAGL_TOOLKIT_REGISTRY,
) -> "Molecule":
    """
    Normalize the bond orders and charges of a molecule by applying a series of transformations to it.

    Parameters
    ----------
    molecule: openff.toolkit.topology.Molecule
        The molecule to normalize.
    max_iter: int, default=200
        The maximum number of iterations to perform for each transformation.
        This parameter is only used with the RDKit ToolkitWrapper,
        as the OpenEyeToolkitWrapper applies each normalization reaction exhaustively.
    inplace: bool, default=False
        If the molecule should be normalized in place or a new molecule should be returned.
        If a new molecule is returned, atoms remain ordered like the original molecule.
    toolkit_registry: openff.toolkit.utils.toolkits.ToolkitRegistry or
        lopenff.toolkit.utils.toolkits.ToolkitWrapper, default=GLOBAL_TOOLKIT_REGISTRY
        :class:`ToolkitRegistry` or :class:`ToolkitWrapper` to use to perform normalization reactions.

    Returns
    -------
    normalized_molecule: openff.toolkit.topology.Molecule
    """
    # normalizations from RDKit's Code/GraphMol/MolStandardize/TransformCatalog/normalizations.in
    normalizations = (
        "[N,P,As,Sb;X3:1](=[O,S,Se,Te:2])=[O,S,Se,Te:3]>>[*+1:1](-[*-1:2])=[*:3]",  # Nitro to N+(O-)=O
        "[S+2:1]([O-:2])([O-:3])>>[S+0:1](=[O-0:2])(=[O-0:3])",  # Sulfone to S(=O)(=O)
        "[nH0+0:1]=[OH0+0:2]>>[n+:1]-[O-:2]",  # Pyridine oxide to n+O-
        "[*:1][N:2]=[N:3]#[N:4]>>[*:1][N:2]=[N+:3]=[N-:4]",  # Azide to N=N+=N-
        "[*:1]=[N:2]#[N:3]>>[*:1]=[N+:2]=[N-:3]",  # Diazo/azo to =N+=N-
        "[!O:1][S+0;X3:2](=[O:3])[!O:4]>>[*:1][S+1:2]([O-:3])[*:4]",  # Sulfoxide to -S+(O-)-
        "[O,S,Se,Te;-1:1][P+;D4:2][O,S,Se,Te;-1:3]>>[*+0:1]=[P+0;D5:2][*-1:3]",  # Phosphate to P(O-)=O
        "[C,S&!$([S+]-[O-]);X3+1:1]([NX3:2])[NX3!H0:3]>>[*+0:1]([N:2])=[N+:3]",  # C/S+N to C/S=N+
        "[P;X4+1:1]([NX3:2])[NX3!H0:3]>>[*+0:1]([N:2])=[N+:3]",  # P+N to P=N+
        # Normalize hydrazine-diazonium
        "[CX4:1][NX3H:2]-[NX3H:3][CX4:4][NX2+:5]#[NX1:6]>>[CX4:1][NH0:2]=[NH+:3][C:4][N+0:5]=[NH:6]",
        # Recombine 1,3-separated charges
        "[N,P,As,Sb,O,S,Se,Te;-1:1]-[A+0:2]=[N,P,As,Sb,O,S,Se,Te;+1:3]>>[*-0:1]=[*:2]-[*+0:3]",
        "[n,o,p,s;-1:1]:[a:2]=[N,O,P,S;+1:3]>>[*-0:1]:[*:2]-[*+0:3]",  # Recombine 1,3-separated charges
        "[N,O,P,S;-1:1]-[a:2]:[n,o,p,s;+1:3]>>[*-0:1]=[*:2]:[*+0:3]",  # Recombine 1,3-separated charges
        # Recombine 1,5-separated charges
        "[N,P,As,Sb,O,S,Se,Te;-1:1]-[A+0:2]=[A:3]-[A:4]=[N,P,As,Sb,O,S,Se,Te;+1:5]>>[*-0:1]=[*:2]-[*:3]=[*:4]-[*+0:5]",  # noqa: E501
        # Recombine 1,5-separated charges
        "[n,o,p,s;-1:1]:[a:2]:[a:3]:[c:4]=[N,O,P,S;+1:5]>>[*-0:1]:[*:2]:[*:3]:[c:4]-[*+0:5]",
        # Recombine 1,5-separated charges
        "[N,O,P,S;-1:1]-[c:2]:[a:3]:[a:4]:[n,o,p,s;+1:5]>>[*-0:1]=[c:2]:[*:3]:[*:4]:[*+0:5]",
        "[N,O;+0!H0:1]-[A:2]=[N!$(*[O-]),O;+1H0:3]>>[*+1:1]=[*:2]-[*+0:3]",  # Normalize 1,3 conjugated cation
        "[n;+0!H0:1]:[c:2]=[N!$(*[O-]),O;+1H0:3]>>[*+1:1]:[*:2]-[*+0:3]",  # Normalize 1,3 conjugated cation
        # Normalize 1,5 conjugated cation
        "[N,O;+0!H0:1]-[A:2]=[A:3]-[A:4]=[N!$(*[O-]),O;+1H0:5]>>[*+1:1]=[*:2]-[*:3]=[*:4]-[*+0:5]",
        # Normalize 1,5 conjugated cation
        "[n;+0!H0:1]:[a:2]:[a:3]:[c:4]=[N!$(*[O-]),O;+1H0:5]>>[n+1:1]:[*:2]:[*:3]:[*:4]-[*+0:5]",
        "[F,Cl,Br,I,At;-1:1]=[O:2]>>[*-0:1]-[O-:2]",  # Charge normalization
        "[N,P,As,Sb;-1:1]=[C+;v3:2]>>[*+0:1]#[C+0:2]",  # Charge recombination
    )

    normalized = call_toolkit_function(
        "_run_normalization_reactions",
        molecule=molecule,
        normalization_reactions=normalizations,
        max_iter=max_iter,
        toolkit_registry=toolkit_registry,
    )  # type: ignore

    if not inplace:
        molecule = copy.deepcopy(molecule)

    for self_atom, norm_atom in zip(molecule.atoms, normalized.atoms):
        self_atom.formal_charge = norm_atom.formal_charge
    for norm_bond in normalized.bonds:
        self_bond = molecule.get_bond_between(
            norm_bond.atom1_index, norm_bond.atom2_index
        )
        self_bond._bond_order = norm_bond.bond_order
    return molecule


def enumerate_stereoisomers(
    molecule: "Molecule",
    undefined_only: bool = True,
    max_isomers: int = 20,
    rationalize: bool = True,
    include_self: bool = False,
    toolkit_registry=NAGL_TOOLKIT_REGISTRY,
) -> List["Molecule"]:
    """Enumerate stereoisomers for a molecule.

    Parameters
    ----------
    molecule
        The molecule to enumerate stereoisomers for.
    undefined_only: bool, optional, default=True
        If we should enumerate all stereocenters and bonds or only those with undefined stereochemistry
    max_isomers: int optional, default=20
        The maximum amount of molecules that should be returned
    rationalize
        If we should try to build and rationalise the molecule to ensure it can exist

    Returns
    -------
        A list of stereoisomers.
    """
    from openff.toolkit.topology import Molecule

    stereoisomers = molecule.enumerate_stereoisomers(
        undefined_only=undefined_only,
        max_isomers=max_isomers,
        rationalise=rationalize,
        toolkit_registry=toolkit_registry,
    )

    if include_self:
        if not any(molecule == isomol for isomol in stereoisomers):
            stereoisomers.append(copy.deepcopy(molecule))
    return stereoisomers


def guess_file_format(file: str) -> str:
    """
    Guess the file format of a file from the extension

    Parameters
    ----------
    file: str
        The file to guess the format of

    Returns
    -------
    str
        The guessed file format
    """
    file = str(file)
    if file.endswith("sdf"):
        file_format = "sdf"
    elif file.endswith("sdf.gz"):
        file_format = "sdf.gz"
    elif file.endswith("smi") or file.endswith("smiles"):
        file_format = "smiles"
    else:
        raise ValueError(f"Unknown file format for file: {file}")
    return file_format


@toolkit_registry_function
def stream_molecules_from_sdf_file(
    file: str,
    explicit_hydrogens: bool = True,
    as_smiles: bool = False,
    mapped_smiles: bool = False,
    include_sdf_data: bool = True,
    toolkit_registry=NAGL_TOOLKIT_REGISTRY,
):
    """Stream molecules from an SDF file.

    Parameters
    ----------
    file
        The path to the SDF file.
    explicit_hydrogens
        Leave explicit hydrogens in the molecules.
    as_smiles
        If True, return the molecules as SMILES strings.
    mapped_smiles
        If True, return mapped smiles with atom indices
    include_sdf_data
        If SDF tag data (e.g. charges) should be included in the molecule properties
    toolkit_registry
        The toolkit registry to use.

    Returns
    -------
        A generator of openff.toolkit.topology.Molecule objects or SMILES strings
    """
    pass


def validate_smiles(smiles: str, toolkit_registry=NAGL_TOOLKIT_REGISTRY):
    from openff.toolkit.topology import Molecule

    offmol = Molecule.from_smiles(smiles, toolkit_registry=toolkit_registry)
    return offmol.to_smiles(
        mapped=False, isomeric=True, toolkit_registry=toolkit_registry
    )


def stream_molecules_from_smiles_file(
    file: str,
    as_smiles: bool = False,
    mapped_smiles: bool = False,
    toolkit_registry=NAGL_TOOLKIT_REGISTRY,
):
    """Stream molecules from a SMILES file.

    Parameters
    ----------
    file
        The path to the SMILES file.
    as_smiles
        If True, return the molecules as SMILES strings.
    mapped_smiles
        If True, return mapped smiles with atom indices
    validate_smiles
        If True, validate SMILES by converting to and from OpenFF Molecule.
        If False, SMILES are assumed to be valid, and `mapped_smiles`
        is ignored.
    toolkit_registry
        The toolkit registry to use.

    Returns
    -------
        A generator of openff.toolkit.topology.Molecule objects or SMILES strings
    """
    from openff.toolkit.topology.molecule import SmilesParsingError
    from openff.toolkit.topology import Molecule

    with open(file, "r") as f:
        smiles = [x.strip() for x in f.readlines()]

    for line in smiles:
        for field in line.split():
            try:
                offmol = Molecule.from_mapped_smiles(
                    field, toolkit_registry=toolkit_registry
                )
            except (ValueError, SmilesParsingError):
                offmol = Molecule.from_smiles(field, allow_undefined_stereo=True)

            if as_smiles:
                offmol = offmol.to_smiles(
                    mapped=mapped_smiles, toolkit_registry=toolkit_registry
                )
            yield offmol


def stream_molecules_from_file(
    file: str,
    file_format: str = None,
    explicit_hydrogens: bool = True,
    as_smiles: bool = False,
    mapped_smiles: bool = False,
    include_sdf_data: bool = True,
    toolkit_registry=NAGL_TOOLKIT_REGISTRY,
):
    """Stream molecules from a file.

    Parameters
    ----------
    file: str
        The path to the file.
    file_format: str, optional
        The file format. If not provided, the format will be guessed from the file extension.
    explicit_hydrogens: bool, optional
        Keep explicit hydrogens if molecule are output as SMILES.
    as_smiles: bool, optional
        If True, return the molecules as SMILES strings.
    mapped_smiles: bool, optional
        If True, return mapped smiles with atom indices
    include_sdf_data: bool, optional
        If SDF tag data (e.g. charges) should be included in the molecule properties
    toolkit_registry
        The toolkit registry to use.

    Returns
    -------
    molecules: Generator[openff.toolkit.topology.Molecule or str]
        A generator of openff.toolkit.topology.Molecule objects or SMILES strings

    """
    if file_format is None:
        file_format = guess_file_format(file)

    if file_format in ("sdf", "sdf.gz"):
        func = functools.partial(
            stream_molecules_from_sdf_file,
            explicit_hydrogens=explicit_hydrogens,
            as_smiles=as_smiles,
            mapped_smiles=mapped_smiles,
            include_sdf_data=include_sdf_data,
            toolkit_registry=toolkit_registry,
        )
    elif file_format == "smiles":
        func = functools.partial(
            stream_molecules_from_smiles_file,
            as_smiles=as_smiles,
            mapped_smiles=mapped_smiles,
            toolkit_registry=toolkit_registry,
        )

    for mol in func(file):
        yield mol


def smiles_to_inchi_key(smiles: str) -> str:
    """Convert a SMILES string to an InChI key.

    Parameters
    ----------
    smiles
        The SMILES string to convert.

    Returns
    -------
    inchi_key
        The InChI key corresponding to the SMILES string.
    """

    from openff.toolkit.topology import Molecule

    offmol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    return offmol.to_inchikey(fixed_hydrogens=True)


def get_openff_molecule_bond_indices(molecule: "Molecule") -> List[Tuple[int, int]]:
    """Get the atom indices of each bond in an OpenFF molecule.

    Parameters
    ----------
    molecule
        The molecule to get the bond indices for.

    Returns
    -------
    bond_indices
        The atom indices of each bond in the molecule.
        The indices are sorted in ascending order for each bond.
        The bonds are ordered by the molecule bond order.
    """
    return [
        tuple(sorted((bond.atom1_index, bond.atom2_index))) for bond in molecule.bonds
    ]


def map_indexed_smiles(reference_smiles: str, target_smiles: str) -> Dict[int, int]:
    """
    Map the indices of the target SMILES to the indices of the reference SMILES.

    Parameters
    ----------
    reference_smiles
        The reference SMILES string, mapped with atom indices.
    target_smiles
        The target SMILES string, mapped with atom indices.

    Returns
    -------
    atom_map
        A dictionary in the form of {reference_atom_index: target_atom_index}
    """
    from openff.toolkit.topology import Molecule

    reference_molecule = Molecule.from_mapped_smiles(
        reference_smiles, allow_undefined_stereo=True
    )
    target_molecule = Molecule.from_mapped_smiles(
        target_smiles, allow_undefined_stereo=True
    )

    _, atom_map = Molecule.are_isomorphic(
        reference_molecule,
        target_molecule,
        return_atom_map=True,
    )
    return atom_map


def molecule_from_networkx(graph):
    from openff.toolkit.topology import Molecule
    
    molecule = Molecule()

    for _, info in graph.nodes(data=True):
        molecule.add_atom(
            atomic_number=info["atomic_number"],
            formal_charge=info["formal_charge"],
            is_aromatic=info["is_aromatic"],
            stereochemistry=info.get("stereochemistry", None),
        )

    for u, v, info in graph.edges(data=True):
        molecule.add_bond(
            u,
            v,
            bond_order=info["bond_order"],
            is_aromatic=info["is_aromatic"],
            stereochemistry=info.get("stereochemistry", None),
        )
    return molecule


def _molecule_to_graph(molecule: "Molecule") -> _MoleculeGraph:
    """
    Convert an OpenFF molecule to a graph representation.
    
    Parameters
    ----------
    molecule
        The molecule to convert.


    Returns
    -------
    graph: _MoleculeGraph
        A graph representation of the molecule.
        This will be a tuple of (atoms, bonds).
        Atoms is a dictionary of integer atom indices to atom information.
        Each atom information dictionary contains the following keys:
        atomic_number, formal_charge, is_aromatic, stereochemistry.
        Bonds is a dictionary of bond indices as a tuple of integers.
        Each bond indices tuple is mapped to bond information.
        Each bond information dictionary contains the following keys:
        bond_order, is_aromatic, stereochemistry.
    """
    atoms = {}
    for i, atom in enumerate(molecule.atoms):
        atoms[i] = {
            "atomic_number": atom.atomic_number,
            "formal_charge": atom.formal_charge,
            "is_aromatic": atom.is_aromatic,
            "stereochemistry": atom.stereochemistry,
        }

    bonds = {}
    for bond in molecule.bonds:
        indices = tuple(sorted((bond.atom1_index, bond.atom2_index)))
        bonds[indices] = {
            "bond_order": bond.bond_order,
            "is_aromatic": bond.is_aromatic,
            "stereochemistry": bond.stereochemistry,
        }

    return _MoleculeGraph(atoms=atoms, bonds=bonds)



def _molecule_from_graph(graph: _MoleculeGraph) -> "Molecule":
    """
    Convert a graph representation to an OpenFF molecule.
    
    Parameters
    ----------
    graph
        The graph representation to convert.
        This should be a tuple of (atoms, bonds).
        Atoms is a dictionary of integer atom indices to atom information.
        Each atom information dictionary contains the following keys:
        atomic_number, formal_charge, is_aromatic, stereochemistry.
        Bonds is a dictionary of bond indices as a tuple of integers.
        Each bond indices tuple is mapped to bond information.
        Each bond information dictionary contains the following keys:
        bond_order, is_aromatic, stereochemistry.
    
    Returns
    -------
    molecule
        The OpenFF molecule representation.
    """
    from openff.toolkit.topology import Molecule

    molecule = Molecule()
    for atom_index in sorted(graph.atoms):
        atom = graph.atoms[atom_index]
        molecule.add_atom(
            atomic_number=atom["atomic_number"],
            formal_charge=atom["formal_charge"],
            is_aromatic=atom["is_aromatic"],
            stereochemistry=atom.get("stereochemistry", None),
        )

    for (u, v), info in graph.bonds.items():
        molecule.add_bond(
            u,
            v,
            bond_order=info["bond_order"],
            is_aromatic=info["is_aromatic"],
            stereochemistry=info.get("stereochemistry", None),
        )
    return molecule