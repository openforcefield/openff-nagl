import importlib
from typing import TYPE_CHECKING

from openff.toolkit.utils.toolkits import RDKIT_AVAILABLE, OPENEYE_AVAILABLE

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule

def _toolkit_wrapper(function_name, *args, **kwargs):
    from openff.toolkit.utils.exceptions import MissingOptionalDependency
    from openff.nagl.utils.openff import openeye, rdkit

    oefunc = getattr(openeye, function_name)
    rdfunc = getattr(rdkit, function_name)
    try:
        return oefunc(*args, **kwargs)
    except MissingOptionalDependency:
        return rdfunc(*args, **kwargs)
    

def normalize_molecule(
    molecule: "Molecule",
    reaction_smarts: List[str],
) -> "Molecule":
    return _toolkit_wrapper("_normalize_molecule")