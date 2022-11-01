import importlib
from typing import TYPE_CHECKING

<<<<<<< HEAD
from openff.toolkit.utils.toolkits import RDKIT_AVAILABLE, OPENEYE_AVAILABLE
=======
from openff.toolkit.utils.toolkits import OPENEYE_AVAILABLE, RDKIT_AVAILABLE
>>>>>>> d93826c465776019fd89c065d4afbe974c014215

if TYPE_CHECKING:
    from openff.toolkit.topology import Molecule

<<<<<<< HEAD
def _toolkit_wrapper(function_name, *args, **kwargs):
    from openff.toolkit.utils.exceptions import MissingOptionalDependency
=======

def _toolkit_wrapper(function_name, *args, **kwargs):
    from openff.toolkit.utils.exceptions import MissingOptionalDependencyError

>>>>>>> d93826c465776019fd89c065d4afbe974c014215
    from openff.nagl.utils.openff import openeye, rdkit

    oefunc = getattr(openeye, function_name)
    rdfunc = getattr(rdkit, function_name)
    try:
        return oefunc(*args, **kwargs)
<<<<<<< HEAD
    except MissingOptionalDependency:
        return rdfunc(*args, **kwargs)
    
=======
    except MissingOptionalDependencyError:
        return rdfunc(*args, **kwargs)

>>>>>>> d93826c465776019fd89c065d4afbe974c014215

def normalize_molecule(
    molecule: "Molecule",
    reaction_smarts: List[str],
) -> "Molecule":
<<<<<<< HEAD
    return _toolkit_wrapper("_normalize_molecule")
=======
    return _toolkit_wrapper("_normalize_molecule")
>>>>>>> d93826c465776019fd89c065d4afbe974c014215
