"""Utilities for working with OpenFF Toolkit molecules."""

import contextlib
import json
from typing import TYPE_CHECKING, Dict, List, Tuple, Union, Optional

import numpy as np
import torch
from openff.utilities import requires_package
from openff.utilities.exceptions import MissingOptionalDependency

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule as OFFMolecule


@requires_package("rdkit")
def _stream_molecules_from_file_rd(file: str):
    from openff.toolkit.topology import Molecule
    from rdkit import Chem

    if file.endswith(".gz"):
        file = file[:-3]

    for rdmol in Chem.SupplierFromFilename(
        file, removeHs=False, sanitize=True, strictParsing=True
    ):
        if rdmol is not None:
            yield Molecule.from_rdkit(rdmol, allow_undefined_stereo=True)

@requires_package("openeye.oechem")
def _copy_sddata_oe(source, target):
    from openeye import oechem
    for dp in oechem.OEGetSDDataPairs(source):
        oechem.OESetSDData(target, dp.GetTag(), dp.GetValue())

@requires_package("openeye.oechem")
def _stream_molecules_from_file_oe(file: str):
    from openeye import oechem

    stream = oechem.oemolistream()
    stream.open(file)
    for oemol in stream.GetOEMols():
        if (stream.GetFormat() == oechem.OEFormat_SDF) and hasattr(oemol, "GetConfIter"):
            for conf in oemol.GetConfIter():
                confmol = conf.GetMCMol()
                _copy_sddata_oe(oemol, confmol)
                _copy_sddata_oe(conf, confmol)
        else:
            confmol = oemol
        yield _stream_conformer_from_oe(confmol)

@requires_package("openeye.oechem")
def _stream_conformer_from_oe(oemol):
    from openff.toolkit.utils.openeye_wrapper import OpenEyeToolkitWrapper
    from openff.toolkit.topology import Molecule

    has_charges = OpenEyeToolkitWrapper._turn_oemolbase_sd_charges_into_partial_charges(oemol)
    offmol = Molecule.from_openeye(oemol, allow_undefined_stereo=True)
    if not has_charges:
        offmol.partial_charges = None
    return offmol


def _stream_molecules_from_file(file: str):
    try:
        for offmol in _stream_molecules_from_file_oe(file):
            yield offmol
    except MissingOptionalDependency:
        for offmol in _stream_molecules_from_file_rd(file):
            yield offmol



def _stream_molecules_from_smiles(file: str):
    from openff.toolkit.topology import Molecule
    
    with open(file, "r") as f:
        contents = f.readlines()
    for line in contents:
        for field in line.split():
            if field:
                try:
                    offmol = Molecule.from_mapped_smiles(field)
                except ValueError:
                    offmol = Molecule.from_smiles(field)
                yield offmol

def get_file_format(file: str, file_format: str = None):
    if file_format is None:
        if file.endswith("sdf"):
            file_format = "sdf"
        elif file.endswith("sdf.gz"):
            file_format = "sdf.gz"
        elif file.endswith("smi") or file.endswith("smiles"):
            file_format = "smi"
    return file_format



def stream_molecules_from_file(
    file: str,
    file_format: Optional[str] = None,
    as_smiles: bool = False,
    mapped: bool = False
):
    file = str(file)

    if not as_smiles:
        returner = lambda x: x
    else:
        returner = lambda x: x.to_smiles(mapped=mapped)

    file_format = get_file_format(file)
    if file_format == "smi":
        reader = _stream_molecules_from_smiles
    else:
        reader = _stream_molecules_from_file
    
    for offmol in reader(file):
        yield returner(offmol)

@requires_package("openeye.oechem")
def openff_to_openeye(molecule):
    from openeye import oechem

    oemol = molecule.to_openeye()
    if molecule.partial_charges is not None:
        partial_charges_list = [
            oeatom.GetPartialCharge() for oeatom in oemol.GetAtoms()
        ]
        partial_charges_str = " ".join([f"{val:f}" for val in partial_charges_list])
        oechem.OESetSDData(oemol, "atom.dprop.PartialCharge", partial_charges_str)
    return oemol



@requires_package("openeye.oechem")
@contextlib.contextmanager
def _stream_molecules_to_file_oe(file: str):  # pragma: no cover
    from openeye import oechem
    from openff.toolkit.topology import Molecule

    stream = oechem.oemolostream(file)

    def writer(molecule: Molecule):
        oechem.OEWriteMolecule(stream, openff_to_openeye(molecule))

    yield writer

    stream.close()


@requires_package("rdkit")
@contextlib.contextmanager
def _stream_molecules_to_file_rd(file: str):
    from rdkit import Chem

    stream = Chem.SDWriter(file)

    def writer(molecule):
        rdmol = molecule.to_rdkit()
        n_conf = rdmol.GetNumConformers()
        if not n_conf:
            stream.write(rdmol)
        else:
            for i in range(n_conf):
                stream.write(rdmol, confId=i)
        stream.flush()

    yield writer

    stream.close()

@contextlib.contextmanager
def stream_molecules_to_file(file: str):
    try:
        with _stream_molecules_to_file_oe(file) as writer:
            yield writer
    except MissingOptionalDependency:
        with _stream_molecules_to_file_rd(file) as writer:
            yield writer


def get_coordinates_in_angstrom(conformer):
    """Get coordinates of conformer in angstrom, without units"""
    from openff.toolkit.topology.molecule import unit as off_unit

    if hasattr(conformer, "m_as"):
        return conformer.m_as(off_unit.angstrom)
    return conformer.value_in_unit(off_unit.angstrom)


def get_unitless_charge(charge, dtype=float):
    """Strip units from a charge and convert to the specified dtype"""
    from openff.toolkit.topology.molecule import unit as off_unit

    return dtype(charge / off_unit.elementary_charge)


def get_openff_molecule_bond_indices(molecule: "OFFMolecule") -> List[Tuple[int, int]]:
    return [
        tuple(sorted((bond.atom1_index, bond.atom2_index))) for bond in molecule.bonds
    ]


def get_openff_molecule_formal_charges(molecule: "OFFMolecule") -> List[float]:
    from openff.toolkit.topology.molecule import unit as off_unit

    # TODO: this division hack should work for both simtk units
    # and pint units. It should probably be removed when we switch to
    # pint only
    return [
        int(atom.formal_charge / off_unit.elementary_charge) for atom in molecule.atoms
    ]


def get_openff_molecule_information(
    molecule: "OFFMolecule",
) -> Dict[str, "torch.Tensor"]:
    charges = get_openff_molecule_formal_charges(molecule)
    atomic_numbers = [atom.atomic_number for atom in molecule.atoms]
    return {
        "idx": torch.arange(molecule.n_atoms, dtype=torch.int32),
        "formal_charge": torch.tensor(charges, dtype=torch.int8),
        "atomic_number": torch.tensor(atomic_numbers, dtype=torch.int8),
    }


def map_indexed_smiles(reference_smiles: str, target_smiles: str) -> Dict[int, int]:
    """
    Map the indices of the target SMILES to the indices of the reference SMILES.
    """
    from openff.toolkit.topology import Molecule as OFFMolecule

    reference_molecule = OFFMolecule.from_mapped_smiles(reference_smiles)
    target_molecule = OFFMolecule.from_mapped_smiles(target_smiles)

    _, atom_map = OFFMolecule.are_isomorphic(
        reference_molecule,
        target_molecule,
        return_atom_map=True,
    )
    return atom_map


def _rd_normalize_molecule(
    molecule: "OFFMolecule",
    reaction_smarts: List[str] = tuple(),
    max_iterations: int = 10000,
) -> "OFFMolecule":

    from openff.toolkit.topology import Molecule as OFFMolecule
    from rdkit import Chem
    from rdkit.Chem import rdChemReactions

    rdmol = molecule.to_rdkit()
    for i, atom in enumerate(rdmol.GetAtoms(), 1):
        atom.SetAtomMapNum(i)

    original_smiles = new_smiles = Chem.MolToSmiles(rdmol)
    has_changed = True

    for smarts in reaction_smarts:
        reaction = rdChemReactions.ReactionFromSmarts(smarts)
        n_iterations = 0

        while n_iterations < max_iterations and has_changed:
            n_iterations += 1
            old_smiles = new_smiles

            products = reaction.RunReactants((rdmol,), maxProducts=1)
            if not products:
                break

            try:
                ((rdmol,),) = products
            except ValueError:
                raise ValueError(
                    f"Reaction produced multiple products: {smarts}")

            for atom in rdmol.GetAtoms():
                atom.SetAtomMapNum(atom.GetIntProp("react_atom_idx") + 1)

            new_smiles = Chem.MolToSmiles(Chem.AddHs(rdmol))
            has_changed = new_smiles != old_smiles

        if n_iterations == max_iterations and has_changed:
            raise ValueError(
                f"{original_smiles} did not normalize after {max_iterations} iterations: "
                f"{smarts}"
            )

    offmol = OFFMolecule.from_mapped_smiles(
        new_smiles, allow_undefined_stereo=True)
    return offmol


def normalize_molecule(
    molecule: "OFFMolecule",
    check_output: bool = True,
    max_iterations: int = 10000,
) -> "OFFMolecule":
    """
    Normalize a molecule by applying a series of SMARTS reactions.
    """
    from openff.toolkit.topology import Molecule as OFFMolecule

    from openff.nagl.data.files import MOLECULE_NORMALIZATION_REACTIONS

    with open(MOLECULE_NORMALIZATION_REACTIONS, "r") as f:
        reaction_smarts = [entry["smarts"] for entry in json.load(f)]

    normalized = _rd_normalize_molecule(
        molecule,
        reaction_smarts=reaction_smarts,
        max_iterations=max_iterations,
    )

    if check_output:
        isomorphic, _ = OFFMolecule.are_isomorphic(
            molecule,
            normalized,
            aromatic_matching=False,
            formal_charge_matching=False,
            bond_order_matching=False,
            atom_stereochemistry_matching=False,
            bond_stereochemistry_matching=False,
        )

        if not isomorphic:
            err = "normalization changed the molecule - this should not happen"
            raise ValueError(err)

    return normalized


@requires_package("rdkit")
def get_best_rmsd(
    molecule: "OFFMolecule",
    conformer_a: np.ndarray,
    conformer_b: np.ndarray,
):
    from rdkit import Chem
    from rdkit.Chem import rdMolAlign
    from rdkit.Geometry import Point3D

    rdconfs = []
    for conformer in (conformer_a, conformer_b):

        rdconf = Chem.Conformer(len(conformer))
        for i, coord in enumerate(conformer):
            rdconf.SetAtomPosition(i, Point3D(*coord))
        rdconfs.append(rdconf)

    rdmol1 = molecule.to_rdkit()
    rdmol1.RemoveAllConformers()
    rdmol2 = Chem.Mol(rdmol1)
    rdmol1.AddConformer(rdconfs[0])
    rdmol2.AddConformer(rdconfs[1])

    return rdMolAlign.GetBestRMS(rdmol1, rdmol2)


def is_conformer_identical(
    molecule: "OFFMolecule",
    conformer_a: np.ndarray,
    conformer_b: np.ndarray,
    atol: float = 1.0e-3,
) -> bool:
    rmsd = get_best_rmsd(molecule, conformer_a, conformer_b)
    return rmsd < atol


def smiles_to_inchi_key(smiles: str) -> str:
    from openff.toolkit.topology import Molecule

    offmol = Molecule.from_smiles(smiles, allow_undefined_stereo=True)
    return offmol.to_inchikey(fixed_hydrogens=True)


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

    warnings.filterwarnings("ignore")

    toolkit_logger = logging.getLogger("openff.toolkit")
    openff_logger_level = toolkit_logger.getEffectiveLevel()
    toolkit_logger.setLevel(logging.ERROR)

    try:
        with capture_oechem_warnings():
            yield
    except MissingOptionalDependency:
        yield

    toolkit_logger.setLevel(openff_logger_level)
