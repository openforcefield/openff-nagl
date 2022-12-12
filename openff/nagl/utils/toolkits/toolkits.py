import importlib
from typing import TYPE_CHECKING

from openff.toolkit.utils.toolkits import OPENEYE_AVAILABLE, RDKIT_AVAILABLE

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule

__all__ = [
    "normalize_molecule",
]


def _toolkit_wrapper(function_name, *args, **kwargs):
    from openff.toolkit.utils.exceptions import MissingOptionalDependencyError

    from openff.nagl.utils.openff import openeye, rdkit

    oefunc = getattr(openeye, function_name)
    rdfunc = getattr(rdkit, function_name)
    try:
        return oefunc(*args, **kwargs)
    except MissingOptionalDependencyError:
        return rdfunc(*args, **kwargs)


def normalize_molecule(
    molecule: "Molecule",
    reaction_smarts: List[str],
) -> "Molecule":
    return _toolkit_wrapper("_normalize_molecule")
