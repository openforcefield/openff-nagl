import functools
import logging
import multiprocessing
from typing import TYPE_CHECKING, Iterable, List, Tuple, Union

from openff.units.elements import MASSES, SYMBOLS

if TYPE_CHECKING:
    from openff.toolkit.topology.molecule import Molecule, unit

logger = logging.getLogger(__name__)

INV_SYMBOLS = {v: k for k, v in SYMBOLS.items()}


def get_atomic_number(el: Union[int, str]) -> int:
    if isinstance(el, int):
        return el
    return INV_SYMBOLS[el]


def apply_filter(
    molecule: "Molecule",
    allowed_elements: Tuple[int],
    min_mass: "unit.Quantity",
    max_mass: "unit.Quantity",
    n_rotatable_bonds: int,
) -> bool:

    mass = sum(MASSES[atom.atomic_number] for atom in molecule.atoms)

    return (
        all(atom.atomic_number in allowed_elements for atom in molecule.atoms)
        and mass > min_mass
        and mass < max_mass
        and len(molecule.find_rotatable_bonds()) <= n_rotatable_bonds
    )


def split_and_apply_filter(
    molecule: "Molecule",
    allowed_elements: Tuple[int],
    min_mass: "unit.Quantity",
    max_mass: "unit.Quantity",
    n_rotatable_bonds: int,
    only_retain_largest: bool = True,
    as_smiles: bool = False,
    mapped_smiles: bool = False
):
    from openff.toolkit.topology import Molecule

    if isinstance(molecule, str):
        if only_retain_largest:
            split = molecule.split(".")
            molecule = max(split, key=len)
        molecule = Molecule.from_smiles(molecule, allow_undefined_stereo=True)
    try:
        if only_retain_largest:
            split_smiles = molecule.to_smiles().split(".")
            if len(split_smiles) > 1:
                largest = max(split_smiles, key=len)
                molecule = Molecule.from_smiles(largest, allow_undefined_stereo=True)
                logger.debug(f"Keeping '{largest}' from '{split_smiles}'")
        valid = apply_filter(
            molecule,
            allowed_elements=allowed_elements,
            min_mass=min_mass,
            max_mass=max_mass,
            n_rotatable_bonds=n_rotatable_bonds,
        )
        if valid:
            if as_smiles:
                return molecule.to_smiles(mapped=mapped_smiles)
            else:
                return molecule
    except Exception as e:
        logger.warning(f"Failed to process molecule {molecule}, {e}")



def filter_molecules(
    molecules: Iterable["Molecule"],
    only_retain_largest: bool = True,
    allowed_elements: Tuple[Union[str, int], ...] = (
        "H",
        "C",
        "N",
        "O",
        "F",
        "P",
        "S",
        "Cl",
        "Br",
        "I",
    ),
    min_mass: "unit.Quantity" = 250,
    max_mass: "unit.Quantity" = 350,
    n_rotatable_bonds: int = 7,
    n_processes: int = 1,
    as_smiles: bool = False,
) -> Iterable["Molecule"]:

    import tqdm
    from openff.toolkit.topology.molecule import unit

    from openff.nagl.utils.openff import capture_toolkit_warnings

    allowed_elements = [get_atomic_number(x) for x in allowed_elements]

    if not isinstance(min_mass, unit.Quantity):
        min_mass = min_mass * unit.amu
    if not isinstance(max_mass, unit.Quantity):
        max_mass = max_mass * unit.amu

    with capture_toolkit_warnings():
        filterer = functools.partial(
            split_and_apply_filter,
            only_retain_largest=only_retain_largest,
            allowed_elements=allowed_elements,
            n_rotatable_bonds=n_rotatable_bonds,
            min_mass=min_mass,
            max_mass=max_mass,
            as_smiles=as_smiles,
        )
        with multiprocessing.Pool(processes=n_processes) as pool:
            for molecule in tqdm.tqdm(
                pool.imap(filterer, molecules), desc="filtering molecules"
            ):
                if molecule is not None:
                    yield molecule
