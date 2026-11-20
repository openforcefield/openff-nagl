from openff.nagl import GNNModel, __version__
from openff.nagl_models import list_available_nagl_models

from openff.toolkit import Molecule, ForceField

assert __version__ != "0.0.0", f"Version check failed! Found {__version__=}"

assert len(list_available_nagl_models()) > 0

[GNNModel.load(model) for model in list_available_nagl_models()]

molecule = Molecule.from_smiles("CC(=O)OC1=CC=CC=C1C(=O)O")

molecule.assign_partial_charges(
    partial_charge_method="openff-gnn-am1bcc-1.0.0.pt",
)

# ForceField("openff-2.3.0.offxml").create_interchange(molecule)
