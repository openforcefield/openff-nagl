import contextlib
<<<<<<< HEAD
from openff.utilities import requires_package

=======

from openff.utilities import requires_package


>>>>>>> d93826c465776019fd89c065d4afbe974c014215
@contextlib.contextmanager
@requires_package("openeye.oechem")
def capture_oechem_warnings():  # pragma: no cover
    from openeye import oechem

    output_stream = oechem.oeosstream()
    oechem.OEThrow.SetOutputStream(output_stream)
    oechem.OEThrow.Clear()

    yield

    oechem.OEThrow.SetOutputStream(oechem.oeerr)


@requires_package("openeye.oechem")
def _normalize_molecule_oe(
    molecule: "Molecule", reaction_smarts: List[str]
) -> "Molecule":  # pragma: no cover

    from openeye import oechem
    from openff.toolkit.topology import Molecule

    oe_molecule: oechem.OEMol = molecule.to_openeye()

    for pattern in reaction_smarts:

        reaction = oechem.OEUniMolecularRxn(pattern)
        reaction(oe_molecule)

    return Molecule.from_openeye(oe_molecule, allow_undefined_stereo=True)
