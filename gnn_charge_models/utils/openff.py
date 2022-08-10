from typing import List, Union, TYPE_CHECKING, Tuple, Dict
import contextlib

import json
import torch
import numpy as np

from openff.utilities import requires_package
from openff.utilities.exceptions import MissingOptionalDependency

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule as OFFMolecule


def get_unitless_charge(charge, dtype=float):
    from openff.toolkit.topology.molecule import unit as off_unit
    return dtype(charge / off_unit.elementary_charge)


def get_openff_molecule_bond_indices(molecule: "OFFMolecule") -> List[Tuple[int, int]]:
    return [
        tuple(sorted((bond.atom1_index, bond.atom2_index)))
        for bond in molecule.bonds
    ]


def get_openff_molecule_formal_charges(molecule: "OFFMolecule") -> List[float]:
    from openff.toolkit.topology.molecule import unit as off_unit
    # TODO: this division hack should work for both simtk units
    # and pint units. It should probably be removed when we switch to
    # pint only
    return [
        int(atom.formal_charge / off_unit.elementary_charge)
        for atom in molecule.atoms
    ]


def get_openff_molecule_information(molecule: "OFFMolecule") -> Dict[str, "torch.Tensor"]:
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

        while (
            n_iterations < max_iterations
            and has_changed
        ):
            n_iterations += 1
            old_smiles = new_smiles

            products = reaction.RunReactants((rdmol,), maxProducts=1)
            if not products:
                break

            try:
                ((rdmol,),) = products
            except ValueError:
                raise ValueError(
                    f"Reaction produced multiple products: {smarts}"
                )

            for atom in rdmol.GetAtoms():
                atom.SetAtomMapNum(atom.GetIntProp("react_atom_idx") + 1)

            new_smiles = Chem.MolToSmiles(Chem.AddHs(rdmol))
            has_changed = new_smiles != old_smiles

        if (
            n_iterations == max_iterations
            and has_changed
        ):
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
    from gnn_charge_models.data.files import MOLECULE_NORMALIZATION_REACTIONS

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
            print("coord", coord)
            rdconf.SetAtomPosition(i, Point3D(*coord))
        rdconfs.append(rdconf)
        print("xxxxxx")

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
